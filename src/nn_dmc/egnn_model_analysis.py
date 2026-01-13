import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------
# Utilities
# ---------------------------
def ensure_float32(t):
    if not torch.is_floating_point(t):
        return t.float()
    return t.to(torch.float32)

# stable sin/cos embedding that accepts (B,N,N) or (B,N,N,1)
def build_sin_cos_embedding(d, n_freqs=4, include_orig=True, clamp_val=1e2):
    """
    d: torch tensor, either shape (B,N,N) or (B,N,N,1)
    Returns: (B,N,N, D_emb) where D_emb = (1 if include_orig else 0) + 2*n_freqs
    """
    if d.dim() == 3:
        d = d.unsqueeze(-1)  # -> (B,N,N,1)
    # frequencies 1..n_freqs
    freq = torch.arange(1, n_freqs + 1, device=d.device, dtype=d.dtype).view(1,1,1,n_freqs)
    angles = d * freq  # (B,N,N,n_freqs)
    angles = torch.clamp(angles, -clamp_val, clamp_val)
    sin_emb = torch.sin(angles)
    cos_emb = torch.cos(angles)
    emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B,N,N,2*n_freqs)
    if include_orig:
        emb = torch.cat([d, emb], dim=-1)  # (B,N,N,1+2*n_freqs)
    return emb

# ---------------------------
# EGNN Layer with LayerNorm + Residuals
# ---------------------------
class EGNNLayerLN(nn.Module):
    def __init__(self, hidden_dim=128, n_freqs=4, coord_rescale=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs
        self.coord_rescale = coord_rescale
        self.d_emb_dim = 1 + 2 * n_freqs

        # message MLP: input = xi || xj || d_emb
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.d_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # coordinate MLP -> scalar per edge
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # feature update MLP (residual)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LayerNorm applied after residual add
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, coords):
        """
        x: (B, N, H)
        coords: (B, N, 3)
        """
        B, N, _ = coords.shape

        # pairwise differences r_ij = r_j - r_i  (B,N,N,3)
        rij = coords.unsqueeze(2) - coords.unsqueeze(1)
        dij = torch.norm(rij, dim=-1, keepdim=True)  # (B,N,N,1)

        # radial embedding (clamped inside)
        d_emb = build_sin_cos_embedding(dij, n_freqs=self.n_freqs, include_orig=True)  # (B,N,N,D)

        # prepare node pair features
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B,N,N,H)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,H)
        edge_in = torch.cat([xi, xj, d_emb], dim=-1)  # (B,N,N,2H+D)

        # messages
        m = self.message_mlp(edge_in)  # (B,N,N,H)

        # coordinate update (scaled residual)
        coord_coef = self.coord_mlp(m).squeeze(-1)  # (B,N,N)
        delta_coords = (coord_coef.unsqueeze(-1) * rij).sum(dim=2)  # (B,N,3)
        coords = coords + self.coord_rescale * delta_coords

        # aggregate messages
        m_sum = m.sum(dim=2)  # (B,N,H)

        # feature update with residual + LayerNorm
        dx = self.feature_mlp(m_sum)  # (B,N,H)
        x = x + dx
        x = self.ln(x)

        return x, coords

# ---------------------------
# Full EGNN model (LayerNorm)
# ---------------------------
class EGNNModelLN(nn.Module):
    def __init__(self, n_atoms, atom_types_list, hidden_dim=128, num_layers=4, n_freqs=4, coord_rescale=0.01):
        super().__init__()
        self.n_atoms = n_atoms

        # store atom types as buffer (1D vector length n_atoms)
        atom_types = torch.as_tensor(atom_types_list, dtype=torch.float32).view(-1)
        self.register_buffer("atom_types", atom_types)

        # atom input embedding: accepts shape (B,N,1)
        self.embedding = nn.Linear(1, hidden_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # stack of layers
        self.layers = nn.ModuleList([
            EGNNLayerLN(hidden_dim=hidden_dim, n_freqs=n_freqs, coord_rescale=coord_rescale)
            for _ in range(num_layers)
        ])

        # output per-atom -> sum
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, coords, z=None):
        """
        coords: (B, n_atoms, 3)
        z: optional per-sample atom types (B, n_atoms, 1). If None, use internal atom_types buffer.
        returns: (B,1)
        """
        B = coords.shape[0]
        if z is None:
            z = self.atom_types.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, N, 1)
        x = self.embedding(z)  # (B,N,H)
        x = self.input_mlp(x)
        for layer in self.layers:
            x, coords = layer(x, coords)
        per_atom = self.out_mlp(x).squeeze(-1)  # (B,N)
        energy = per_atom.sum(dim=1, keepdim=True)  # (B,1)
        return energy

def egnn_cart_to_pot(system, ckpt_file, coords_np, batch_size=1024):
    """
    Optimized CPU evaluation of geometries in batches.
    coords_np: np.ndarray of shape (N,17,3)
    Returns: np.ndarray of energies in a.u.
    """

    if system == 'h11o6':
        atom_types = torch.tensor([8.0, 1.0, 1.0, 8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0], dtype=torch.float32)
        n_atoms = 17
    elif system == 'h9o5':
        atom_types = torch.tensor([8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0, 1.0,
    8.0, 1.0, 1.0, 8.0, 1.0], dtype=torch.float32)
        n_atoms = 14

    model = EGNNModelLN(n_atoms=n_atoms, atom_types_list=atom_types, hidden_dim=64, num_layers=3, n_freqs=3, coord_rescale=0.01)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu = ckpt["mu"]
    sigma = ckpt["sigma"]

    # Pre-allocate atom types tensor

    _atom_types_tensor = atom_types.unsqueeze(0).unsqueeze(-1)  # (1,17,1)

    N_total = coords_np.shape[0]
    energies = np.empty((N_total,), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, N_total, batch_size):
            end = min(start + batch_size, N_total)
            batch = torch.tensor(coords_np[start:end], dtype=torch.float32)  # (B,17,3)
            B = batch.shape[0]

            # expand atom types once per batch
            z = _atom_types_tensor.expand(B, -1, -1)  # (B,17,1)

            y_norm = model(batch, z).squeeze(-1)  # (B,)

            # store energies in-place
            energies[start:end] = (y_norm * sigma + mu).numpy()

    return energies  # convert to a.u.

def calc_egnn_test_errors(system, ckpt_file,cds_data,energy_data):

    cds_test = np.load(cds_data)
    energies_test = np.load(energy_data)

    output = egnn_cart_to_pot(system, ckpt_file, cds_test)

    test_errors = output - energies_test

    average_error = np.mean(test_errors)

    test_MAE = np.mean(np.abs(test_errors))

    return energies_test, output, test_MAE, average_error


def plot_egnn_2d_pred_errors(system,ckpt_file,cds_data,energy_data):
    if system == 'h11o6':
        bin_width = 1400
        bin_height = 120
        xlim = 60000
        ylim = 2000
        x_ticks = np.arange(0,75000,15000)
    elif system == 'h9o5':
        bin_width = 1200
        bin_height = 120
        xlim = 50000
        ylim = 2000
        x_ticks = None
    
    energies_test, output, test_MAE, average_error = calc_egnn_test_errors(system,ckpt_file,cds_data,energy_data)

    test_errors = output - energies_test

    fig, ax = plt.subplots()

    h = ax.hist2d(energies_test,test_errors,bins=[np.arange(0,xlim+bin_width,bin_width),np.arange(-ylim,ylim+bin_height,bin_height)],norm=LogNorm(vmin=1),cmap = 'viridis')

    fig.colorbar(h[3], ax=ax)

    plt.hlines(0,0,xlim,color = 'white',linewidth = 1.5)

    steps = np.arange(0,41,3)

    error_pairs = []
    for i in range(len(test_errors)):
        error_pairs.append([energies_test[i],test_errors[i]])
        
    bins = []
    for i in range(len(h[1])-1):
        bin_elements = []
        for pair in error_pairs:
            if h[1][i] < pair[0] and h[1][i+1] > pair[0]:
                bin_elements.append(pair[1])
        bins.append(bin_elements)
        
    bin_stats = []
    for k in bins:
        bin_stats.append([np.mean(k),np.std(k)])

    for j in steps:
        #mid_bin = (h[1][steps[j]]+h[1][steps[j+1]])/2
        plt.vlines(h[1][j],bin_stats[j][0]-bin_stats[j][1],bin_stats[j][0]+bin_stats[j][1],color = 'magenta',linewidth = 3)
        plt.scatter(h[1][j],bin_stats[j][0],color = 'magenta')

    j = steps[7]
    plt.vlines(h[1][j],bin_stats[j][0]-bin_stats[j][1],bin_stats[j][0]+bin_stats[j][1],color = 'black',linewidth = 3)
    plt.scatter(h[1][j],bin_stats[j][0],color = 'black')
        
    plt.xlim(0,xlim)
    plt.xlabel('E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylabel(r'E(NN) - E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylim(-ylim,ylim)
    
    plt.xticks(x_ticks)

    plt.show()

    print('MAE: {0:0.2f}, average error: {1:0.2f}'.format(test_MAE,average_error))


def plot_egnn_pred_errors(ckpt_file,cds_data,energy_data):

    y_test_regular, output_regular, _, _ = calc_egnn_test_errors(ckpt_file,cds_data,energy_data)

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5, 5))
    x = np.linspace(0,100000,1000)
    plt.plot(x,x,linestyle = 'dashed',color = 'black')
    plt.scatter(y_test_regular,output_regular,color = 'rebeccapurple')
    plt.xlabel(r'E(MOB-ML) (/cm$^{-1}$)',fontsize=16)
    plt.ylabel(r'E(NN) (/cm$^{-1}$)',fontsize=16)
    plt.xlim(0,100000)
    plt.ylim(0,100000)

    plt.xticks(np.arange(0, 120000, 20000))
    plt.yticks(np.arange(0, 120000, 20000))

    plt.show()