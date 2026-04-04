import numpy as np
import matplotlib.pyplot as plt
import os

from my_functions import generalized_reconstruction


def create_grid(motif, exponent_size, row_offset=0, col_offset=0):
    """Place motif into a zero-padded grid of size 2^exponent_size.

    Args
    ---
    motif: motif to place
    exponent_size: int
        grid size = 2^exponent_size
    Returns
    --
    grid : grid with motif inserted
    """
    grid_size = 2**exponent_size
    grid = np.zeros((grid_size, grid_size))

    h, w = motif.shape

    # end positions
    r_end = row_offset + h
    c_end = col_offset + w

    grid[row_offset:r_end, col_offset:c_end] = motif
    return grid


def plot_save_grid(grid, title=None, filename=None, cell_size=2, show_ticks=True):
    """Plot grid, optionally saves to plots/.

    Args
    ---
    cell_size: int, optional
        cell size for grid lines. Default 2
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)

    if title:
        ax.set_title(title, fontsize=15)

    point_indices = np.arange(grid.shape[0])
    ax.set_xticks(point_indices)
    ax.set_yticks(point_indices)

    # gridlines
    cell_boundaries = np.arange(-0.5, grid.shape[0], cell_size)
    ax.set_xticks(cell_boundaries, minor=True)
    ax.set_yticks(cell_boundaries, minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=1, alpha=1)

    ax.tick_params(
        which="major",
        bottom=show_ticks,
        left=show_ticks,
        labelbottom=show_ticks,
        labelleft=show_ticks,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    if filename:
        full_path = os.path.join("plots", filename)
        plt.savefig(full_path, dpi=600, bbox_inches="tight")

    plt.show()


def create_superposition_mps(*mps_list):
    """
    Superpose multiple product-state MPS into one, compressing shared suffixes.

    Function works as follows:
    BACKWARD PASS (right to left)
    - find which MPS share identical suffixes
    - store which bond node each MPS occupies at each site

    FORWARD PASS (left to right)
    - build compressed tensors using node assignments from above
    - accumulate coefficients at site 0 where mps paths are the same


    Args
    --
    *mps_list:
        any number of equal-length MPS (bond dim 1 assumed)

    Returns
    ---
    new_mps: list
        compressed sum MPS
    """

    L = len(mps_list[0])
    num_states = len(mps_list)

    # assign coordinates per bond position
    node_at_bond = [None] * (L + 1)
    node_at_bond[L] = [0] * num_states

    # BACKWARD PASS
    for site in range(L - 1, 0, -1):
        unique_paths = {}
        next_available_node = 0

        for k, mps in enumerate(mps_list):
            # path = (tensor values, right connection)
            tensor_key = (tuple(mps[site].flatten()), mps[site].shape)
            path = (tensor_key, node_at_bond[site + 1][k])

            # same node for identical paths
            if path not in unique_paths:
                unique_paths[path] = next_available_node
                next_available_node += 1

            node_at_bond[site] = node_at_bond[site] or []
            node_at_bond[site].append(unique_paths[path])

    # BC condition, all mps start on same note at first site
    node_at_bond[0] = [0] * num_states

    # FORWARD PASS
    new_mps = []
    for site in range(L):
        left_dim = max(node_at_bond[site]) + 1  # unique nodes on left bond
        right_dim = max(node_at_bond[site + 1]) + 1  # unique nodes on right bond
        phys_dim = mps_list[0][site].shape[1]

        new_tensor = np.zeros(
            (left_dim, phys_dim, right_dim), dtype=mps_list[0][site].dtype
        )

        if site == 0:
            # accumulate all
            for k, mps in enumerate(mps_list):
                left = node_at_bond[site][k]
                right = node_at_bond[site + 1][k]
                new_tensor[left, :, right] += mps[site].squeeze()
        else:
            # store unique tensors
            already_filled = set()
            for k, mps in enumerate(mps_list):
                left = node_at_bond[site][k]
                right = node_at_bond[site + 1][k]

                if (left, right) not in already_filled:
                    new_tensor[left, :, right] = mps[site].squeeze()
                    already_filled.add((left, right))

        new_mps.append(new_tensor)

    return new_mps


def plot_motif_reconstruction(
    lattice_mps,
    lambdas,
    depth,
    show_values=True,
    title=None,
    grid_step=None,
    target_axes=None,
    return_raw_data=False,
    filename=None,
):
    """
    Reconstruct and plot motifs from a given MPS depth.

    Args
    --
    lattice_mps:
        MPS tensor list
    lambdas: list
        singular value arrays, to ensure correct pixel scaligng
    depth: int
        depth index
    show_values: optional
        overlay pixel values. Default True.
    grid_step: int, optional
        white gridline spacing; no grid if None
    return_raw_data: bool, optional
        if True, also returns motifs_lattice array

    Returns
    --
    plot_obj :
        Figure if target_axes is None, else list of plot objects
    motifs_lattice : ndarray
        only returned if return_raw_data=True.
    """
    # example for 5 site mps
    # depth=3: last site,
    # depth=0: full chain
    size_for_recon = len(lattice_mps) - depth
    size_for_recon = max(1, min(size_for_recon, len(lattice_mps)))

    last_site = lattice_mps[-size_for_recon:]
    num_features = last_site[0].shape[0]

    motifs_lattice = []
    for f in range(num_features):
        img = generalized_reconstruction(last_site, selection_index=f)
        motifs_lattice.append(img)

    # apply norm
    motifs_lattice = np.array(motifs_lattice) * lambdas[-1]
    mot_min, mot_max = np.min(motifs_lattice), np.max(motifs_lattice)
    n_motifs = len(motifs_lattice)

    if target_axes is not None:
        axes_flat = np.array(target_axes).flatten()
    else:
        fig, axes = plt.subplots(1, n_motifs, figsize=(7 * n_motifs, 5), squeeze=False)
        axes_flat = axes.flatten()
        if title is None:
            plot_center_x = 0.45
            fig.suptitle(
                f"Motif for Lattice (Depth {depth})", fontsize=16, x=plot_center_x
            )
        elif title is not False:
            plot_center_x = 0.45
            fig.suptitle(title, fontsize=16, x=plot_center_x)
        plot_obj = fig

    ims = []
    for i, ax in enumerate(axes_flat):
        if i >= len(motifs_lattice):
            break

        motif = motifs_lattice[i]
        h, w = motif.shape

        im = ax.imshow(motif, cmap="gray", vmin=mot_min, vmax=mot_max)
        ims.append(im)

        if grid_step is not None:
            x_pos = np.arange(-0.5, w + 0.01, grid_step)
            y_pos = np.arange(-0.5, h + 0.01, grid_step)

            ax.vlines(
                x_pos, ymin=-0.5, ymax=h - 0.5, colors="black", linewidth=2, zorder=10
            )
            ax.hlines(
                y_pos, xmin=-0.5, xmax=w - 0.5, colors="black", linewidth=2, zorder=10
            )

            ax.vlines(
                x_pos, ymin=-0.5, ymax=h - 0.5, colors="white", linewidth=1, zorder=11
            )
            ax.hlines(
                y_pos, xmin=-0.5, xmax=w - 0.5, colors="white", linewidth=1, zorder=11
            )

        ax.grid(False)

        if show_values:
            for y in range(h):
                for x in range(w):
                    val = motif[y, x]
                    text_color = "white" if val < (mot_max + mot_min) / 2 else "black"
                    ax.text(
                        x,
                        y,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                        fontweight="bold",
                    )

        ax.tick_params(
            which="major", bottom=False, left=False, labelbottom=False, labelleft=False
        )

        if target_axes is not None:
            if title is False:
                pass
            elif title is not None:
                plot_label = f"{title} {i}" if n_motifs > 1 else title
                ax.set_title(plot_label, fontsize=14)
            else:
                ax.set_title(f"Motif {i}")

    if target_axes is None:
        fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.90, wspace=0.3)
    else:
        plot_obj = ims

    # save
    if filename and target_axes is None:
        full_path = os.path.join("plots", filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")

    if return_raw_data:
        return plot_obj, motifs_lattice

    return plot_obj
