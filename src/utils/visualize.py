import torch
from ase.spacegroup import crystal
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

from src.utils.elements import elements
from src.models.operator.utils import lattice_params_to_matrix_torch


def select(idx, cell, pos, z, num_atoms):
    struct_idx = torch.arange(num_atoms.shape[0], device=num_atoms.device)
    batch = struct_idx.repeat_interleave(num_atoms)

    mask = idx == batch

    return (
        cell[idx].clone().detach().cpu().numpy(),
        pos[mask].clone().detach().cpu().numpy(),
        z[mask].argmax(dim=1).clone().detach().cpu().numpy(),
    )


def plot(ax, idx, cell, pos, z, num_atoms, radii=0.3, rotation=("90x,45y,0z")):
    cell, pos, z = select(idx, cell, pos, z, num_atoms)
    c = crystal(z, basis=pos, cell=cell)
    plot_atoms(c, ax, radii=radii, rotation=rotation)


def plot_grid(
    cell, pos, z, num_atoms, rows=2, cols=3, radii=0.3, rotation=("30x,30y,30z")
):
    fig, axs = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            plot(
                axs[i][j],
                j + i * cols,
                cell,
                pos,
                z,
                num_atoms,
                radii=radii,
                rotation=rotation,
            )

    return fig


def generate_fig(original, denoised, n):
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ase.spacegroup import crystal

    L_o, x_o, z_o, atoms_count_o = original
    batch_o = torch.arange(L_o.shape[0], device=L_o.device)
    batch_atoms_o = batch_o.repeat_interleave(atoms_count_o)

    L_t, x_t, z_t, atoms_count_t = denoised
    batch_t = torch.arange(atoms_count_t.shape[0], device=atoms_count_t.device)
    batch_atoms_t = batch_t.repeat_interleave(atoms_count_t)

    elems = ["" for _ in range(128)]
    for s, e in elements.items():
        elems[e] = s

    fig, axarr = plt.subplots(n, 2, figsize=(15, n * 5))

    for i in range(n):
        mask_o = batch_atoms_o == i
        mask_t = batch_atoms_t == i

        for k, (L, x, z, title) in enumerate(
            zip(
                [L_o[i], L_t[i]],
                [x_o[mask_o], x_t[mask_t]],
                [z_o[mask_o], z_t[mask_t]],
                ["original", "denoised"],
            )
        ):

            cell_i = L.clone().detach().cpu().numpy()
            x_i = x.clone().detach().cpu().numpy()
            z_i = z.clone().detach().cpu().numpy()

            sym_i = [elems[max(e, 1)] for e in z_i]

            cry = crystal(sym_i, [tuple(x) for x in x_i], cell=cell_i)

            axarr[i][k].set_title(title)
            axarr[i][k].set_axis_off()
            try:
                plot_atoms(cry, axarr[i][k], rotation=("45x,45y,0z"))
            except:
                pass

    # fig.savefig(os.path.join(output_directory, f"gen_{n_iter}.png"))
    return fig


def get_fig(batch, model, n, lattice_scaler=None):

    # get data from the batch
    L_real = batch.cell
    x_real = batch.pos
    z_real = batch.z
    struct_size = batch.num_atoms

    # denoise
    L_denoised = model(L_real, x_real, z_real, struct_size)

    if isinstance(L_denoised, tuple):
        lengths_scaled, angles_scaled = lattice_scaler.denormalise(
            L_denoised[0], L_denoised[1])
        L_denoised = lattice_params_to_matrix_torch(
            lengths_scaled, angles_scaled)

    original = (L_real, x_real, z_real, struct_size)
    denoised = (L_denoised, x_real, z_real, struct_size)

    return generate_fig(original, denoised, n)
