import torch
import numpy as np
from h11o6_egnn_training import EGNNModelLN

# ---------------------------------------------------------
#  Detect device (GPU if available)
# ---------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EGNN Potential] Using device: {_DEVICE}")

# ---------------------------------------------------------
#  Load model ONCE globally
# ---------------------------------------------------------

atom_types = torch.tensor(
    [8.0,1.0,1.0,8.0,1.0,1.0,
     8.0,1.0,1.0,8.0,1.0,1.0,
     8.0,1.0,1.0,8.0,1.0],
    dtype=torch.float32
)

# Create model
model = EGNNModelLN(
    n_atoms=17,
    atom_types_list=atom_types,
    hidden_dim=64,
    num_layers=3,
    n_freqs=3,
    coord_rescale=0.01,
)

# Load checkpoint
ckpt = torch.load("h11o6_egnn_3_2_64_3_3_ln_best.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
mu = ckpt["mu"]
sigma = ckpt["sigma"]

# Move model to device
model.to(_DEVICE)
model.eval()

print("[EGNN Potential] Model loaded + moved to device.")

# ---------------------------------------------------------
#  Preallocated buffers
# ---------------------------------------------------------
_MAX_BATCH = 40000

# atom types buffer (max batch)
_z_buf = atom_types.unsqueeze(0).unsqueeze(-1).expand(_MAX_BATCH, -1, -1).clone().to(_DEVICE)

# coordinate buffer
_coords_buf = torch.zeros((_MAX_BATCH, 17, 3), dtype=torch.float32, device=_DEVICE)

# cm⁻¹ → Hartree conversion
CM_TO_H = 1.0 / 219474.63136320


# ---------------------------------------------------------
#  Main evaluation function (callable by PyVibDMC)
# ---------------------------------------------------------

def cart_to_pot(coords_np, batch_size=4096):
    """
    coords_np: np.ndarray (N,17,3)
    Returns: energies in Hartree
    """

    N_total = coords_np.shape[0]
    energies = np.empty(N_total, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, N_total, batch_size):
            end = min(start + batch_size, N_total)
            B = end - start

            # copy coords into GPU tensor
            _coords_buf[:B].copy_(torch.from_numpy(coords_np[start:end]))

            # slice buffers
            batch = _coords_buf[:B]
            z = _z_buf[:B]

            # forward pass on chosen device
            y_norm = model(batch, z).squeeze(-1)    # (B,)

            # unnormalize
            y = (y_norm * sigma + mu) * CM_TO_H     # convert to Hartree

            # bring to CPU numpy
            energies[start:end] = y.cpu().numpy()

    return energies