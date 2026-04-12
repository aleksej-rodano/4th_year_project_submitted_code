import itertools
import numpy as np
from joblib import Parallel, delayed

from ncon import ncon
from scipy.linalg import svd
import matplotlib.pyplot as plt

# function taken from Ivan Shalashilinsky code


def celg2(k):
    return int(np.ceil(np.log2(k)))


H = 1 / 2 * np.array([[1, 1], [1, -1]])


def n_hadamard(n):
    mat_temp = H
    for i in range(n - 1):
        mat_temp = np.kron(mat_temp, H)
    return mat_temp


def classify_state(
    classifier_mpo, mps_cluster, n_classes, Z0, class_labels, classify_type="classical"
):
    """
    computes the inner product between MPO (classifier) and state image.
    Args:
        classifier_mpo: list of MPO tensors
        mps_cluster: list of MPS tensors (batched)
        n_classes: number of classes
        Z0: whether to perform reweighting
        class_labels: list of class labels
        classify_type: 'classical' or 'quantum'

    Returns:
        outcome: array of shape (dataset_size, n_classes) with classification probabilities
        Z_outcomes: array of shape (dataset_size, 2**n_qubits_label) with Z-expectation values
        states_opped: only for quantum classification, the states after applying the classifier


    """
    # n_classes = classifier_mpo[0].shape[0]
    n_sites = len(mps_cluster)
    dataset_size = len(mps_cluster[0])

    # conjugate everything in the MPS (is this necessary with real numbers)

    # outcome_total = np.zeros((dataset_size, n_classes))

    # for k, mps in tqdm(enumerate(mps_cluster)):

    # contract along the physical dimension first
    # think about this tomorrow: which is faster?

    # first contract along real indices
    outcome_temp = []
    outcome = None
    if classify_type == "classical":
        for i in range(n_sites):
            site_inner_temp = np.einsum(
                "ijk,Nljm->Nilkm", classifier_mpo[i], mps_cluster[i].conj()
            )
            N, I, J, K, L = site_inner_temp.shape
            outcome_temp.append(site_inner_temp.reshape(N, I * J, K * L))

            # print([outcome_temp[l].shape for l in range(len(outcome_temp))])

        outcome = outcome_temp[0]
        for i in range(n_sites - 1):
            outcome = np.einsum("Nij, Njk -> Nik", outcome, outcome_temp[i + 1])

        # outcome = outcome.reshape(N, n_classes) ** 2

        # return outcome, None, None

        # Capture the raw overlap (inner product)
        outcome_prob = outcome.reshape(N, n_classes) ** 2

        # Prepare Z_outcomes (Padding + Hadamard Transform)
        # Calculate number of qubits needed (e.g., ceil(log2(10)) = 4 -> 16 dims)
        n_qubits = celg2(n_classes)
        # Ensure at least 1 qubit to avoid dimension mismatch if n_classes=1
        if n_qubits == 0:
            n_qubits = 1
        dim_full = 2**n_qubits

        # Pad with zeros
        prob_padded = np.zeros((N, dim_full))
        prob_padded[:, :n_classes] = outcome_prob

        # Apply Hadamard transform to get Z-expectations
        # P = H * Z  =>  Z = H * P (since H is symmetric and self-inverse)
        H_mat = n_hadamard(n_qubits)
        Z_outcomes = prob_padded @ H_mat

        # Return outcome_prob (for accuracy check) and Z_outcomes (for reweighting)
        return outcome_prob, Z_outcomes, None


def batch_mps_cluster(mps_cluster):
    """Data processes. Takes list of MPS and groups them into batches

    Args:
        mps_cluster: list of MPS tensors

    Returns:
        mps_cluster_batched: list of MPS tensors with an added batch dimension"""
    dataset_size = len(mps_cluster)
    n_sites = len(mps_cluster[0])
    return [
        np.stack([mps_cluster[k][i] for k in range(dataset_size)], axis=0)
        for i in range(n_sites)
    ]


# split_L_canonical and dense_to_mps_L_canonical are adapted from https://pennylane.ai/qml/demos/tutorial_mps
# by Korbinian Kottmann


def split_L_canonical(M, physical_dim, bond_dim):
    """
    SVD split of M with truncation to bond_dim, L-canonical form.

    Args
    ---
    M: 2D array
        matrix to decompose.
    physical_dim: int
        physical size
    bond_dim: int
        max bond dimention

    Returns
    ---
    U: 3D array
        left tensor, shape (left_bond, physical_dim, chi)
    Lambda: 1D array
        singular values, truncated to chi
    Vd: 3D array
        right tensor, shape (chi, physical_dim, right_bond)
    """
    U, Lambda, Vd = svd(M, full_matrices=False)

    bonds = len(Lambda)

    # Vd -> (left_bond, phys, right_fused)
    Vd = Vd.reshape(bonds, physical_dim, -1)

    # U -> (left_bond, phys, right_bond)
    U = U.reshape((-1, physical_dim, bonds))

    # chi = min(available bonds, requested bond_dim)
    chi = np.min([bonds, bond_dim])
    # truncate bond index
    U, Lambda, Vd = (
        U[:, :, :chi],
        Lambda[:chi],
        Vd[:chi, :, :],
    )
    return U, Lambda, Vd


def dense_to_mps_L_canonical(psi, physical_dim, bond_dim):
    """
    Convert dense state tensor psi to L-canonical MPS via iterative SVD.

    Args
    ---
    psi: ndarray
        state tensor, matrix or higher
    physical_dim: int
        physical size
    bond_dim: int
        max bond dimension.

    Returns
    ---
    Ms:list of ndarray
        MPS tensors, each (bond_left, physical_dim, bond_right)
    Lambdas: list of ndarray
        singular values at each bond
    """
    Ms = []
    Lambdas = []

    # number of physical sites
    psi_order = psi.ndim

    if psi_order > 2:
        # flatten all but first leg: psi[2, 2, 2..] -> psi[2, (2x2x2...)]
        psi = psi.reshape(psi.shape[0], -1)

    U, Lambda, Vd = split_L_canonical(
        psi, physical_dim, bond_dim
    )  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    Lambdas.append(Lambda)
    bond_left = Vd.shape[0]

    # absorb Lambda into Vd: Lambda[chi] Vd[chi, phys, rem] into psi[chi, phys, rem]
    psi = np.tensordot(np.diag(Lambda), Vd, axes=([1], [0]))

    # loop psi_order-2 times (first and last sites modified separately)
    for _ in range(psi_order - 2):
        # fuse left bond and current physical index
        psi = psi.reshape(
            bond_left * physical_dim, -1
        )  # reshape psi[2 * bond_left, (2x2x2...)]

        U, Lambda, Vd = split_L_canonical(
            psi, physical_dim, bond_dim
        )  # psi[bond_dim x s2, (2x2x..)] = U[(mu1 s2), mu2] S[mu2] Vd[mu2, (2x2x2x..)]

        Ms.append(U)
        Lambdas.append(Lambda)
        # absorb remainder
        psi = np.tensordot(np.diag(Lambda), Vd, axes=([1], [0]))
        bond_left = Vd.shape[0]

    # last site: trivial SVD, Lambda = norm of psi (= 1 if normalised)
    psi = psi.reshape(-1, 1)
    U, Lambda, _ = svd(psi, full_matrices=False)

    # add right dummy index
    U = U.reshape(-1, physical_dim, 1)
    Ms.append(U)
    Lambdas.append(Lambda)
    return Ms, Lambdas


# ------------
# All the following functions are new, written by me
# ------------


def image_to_hierarchical(image_array):
    """
    Reshape NxN image (N = 2^n) into hierarchical (4,...,4) array

    Args
    ---
    image_array: 2D ndarray
        square image, N a power of 2

    Returns
    ---
    hierarchical_arr: ndarray
        shape (4,)*n, where N = 2^n
    """
    N = image_array.shape[0]
    # n = log2(N), one index per bit
    n = int(np.log2(N))

    # (N, N) into (2,)*2n with axes (y_n-1,...,y_0, x_n-1,...,x_0)
    new_shape = (2,) * (2 * n)
    reshaped_arr = image_array.reshape(new_shape)

    transpose_axes = []
    for i in range(n):
        transpose_axes.append(i)  # y_index
        transpose_axes.append(i + n)  # x_index

    # transpose into (y_{n-1}, x_{n-1}, ..., y_0, x_0)
    hierarchical_arr = reshaped_arr.transpose(transpose_axes)

    final_shape = (4,) * (n)
    hierarchical_arr = hierarchical_arr.reshape(final_shape)

    return hierarchical_arr


def generalized_reconstruction(tensor_chain, selection_index=0, transpose_result=False):
    """
    Reconstruct 2D image from an MPS/MPO tensor chain.

    Contracts chain, unpacks (4,...,4) to (2,...,2), transposes to align x/y.
    Coords are encoded as (j,i) by default; use transpose_result=True for (Row,Col) plotting

    note: Due to the coordinate encoding in image_to_vector (which encodes as (j,i)
    instead of (i,j)), the reconstructed image is in (Col, Row) format by default.
    Use transpose_result=True to get the standard (Row, Col) orientation for plotting


    Args
    ---
    tensor_chain:list
        MPS tensors for the chain
    selection_index: int, optional
        which class slice to reconstruct (0 for single MPS). Default 0
    transpose_result: bool
        if True, returns image.T in (Row,Col) orientation

    Returns
    ---
    image:2D ndarray
        reconstructed image
    """

    # root tensor
    root_tensor = tensor_chain[0][selection_index]

    # contract down the chain
    current_tensor = root_tensor
    for next_tensor in tensor_chain[1:]:
        current_tensor = np.tensordot(current_tensor, next_tensor, axes=([-1], [0]))

    # remove dummy index
    state_compact = np.squeeze(current_tensor)

    total_elements = state_compact.size
    num_dims = int(np.log2(total_elements))

    # unpack (4,...,4) to (2,...,2)
    unpacked_shape = (2,) * (num_dims)
    unpacked_state = state_compact.reshape(unpacked_shape)

    # get y and x indices
    y_indices = list(range(0, num_dims, 2))
    x_indices = list(range(1, num_dims, 2))

    transpose_axes = y_indices + x_indices
    image_tensor = unpacked_state.transpose(transpose_axes)

    # reshape into 2D grid
    depth = num_dims // 2
    side_length = 2**depth
    final_image = image_tensor.reshape(side_length, side_length)

    if transpose_result:
        return np.real(final_image).T
    else:
        return np.real(final_image)


def left_to_right_sweep(mps, depth=None):
    """
    Left to right gauge sweep of an MPS

    Args
    ---
    mps : list
        rank-3 tensors (left, phys, right)
    depth: int, optional
        sweep up to this site index; defaults to last bond

    Returns
    ---
    mps: list
        MPS in left canonical gauge
    Lambda: 1D array
        singular values at the last swept bond
    W: 2D array
        gauge transform at the last site.
    """
    As_new = [tensor.copy() for tensor in mps]
    num_sites = len(As_new)

    max_depth = num_sites - 1
    if depth is None:
        sweep_end = max_depth
    else:
        sweep_end = min(depth, max_depth)

    # trivial 1 left env
    env = np.eye(As_new[0].shape[0])

    for i in range(sweep_end):
        An = As_new[i]

        # density matrix on right bond
        L = ncon([env, An.conj(), An], [[1, 2], [1, 3, -1], [2, 3, -2]])

        W, Lambda, W_dagger = svd(L, full_matrices=False)

        # update current site
        As_new[i] = ncon([An, W], [[-1, -2, 1], [1, -3]])

        # push gauge into next site
        As_new[i + 1] = ncon([W_dagger, As_new[i + 1]], [[-1, 1], [1, -2, -3]])

        # update env
        env = ncon([env, As_new[i].conj(), As_new[i]], [[1, 2], [1, 3, -1], [2, 3, -2]])

    return As_new, Lambda, W


def evaluate_accuracy(model, test_clusters, ensemble_mode=False):
    """
    Accuracy evaluation for either a full MPO or an ensemble of MPS chains.

    Options:
    ensemble_mode=False: single MPO, one classify_state call with n_classes.
    ensemble_mode=True: list of 10 MPS chains, one call per chain, scores stacked

    Args
    ---
    model :
        MPO tensor list (ensemble_mode=False) or list of 10 MPS chains (True)
    test_clusters: list of lists
        MPS states per class
    ensemble_mode: default False
        select if model is a single MPO or an ensemble of MPS chains

    Returns
    ---
    class_accuracies: list
        per-class accuracy
    overall_acc: float
        overall accuracy
    """
    num_classes = len(test_clusters)

    # flatten all clusters into one batch
    all_images = []
    cluster_sizes = []
    label_map = []
    for label_idx, cluster in enumerate(test_clusters):
        if len(cluster) == 0:
            continue
        all_images.extend(cluster)
        cluster_sizes.append(len(cluster))
        label_map.append(label_idx)

    all_batched = batch_mps_cluster(all_images)

    # compute scores
    if ensemble_mode:
        all_scores = []
        for chain in model:
            scores, _, _ = classify_state(
                chain,
                all_batched,
                n_classes=1,
                Z0=True,
                class_labels=None,
                classify_type="classical",
            )
            all_scores.append(scores.flatten())
        outcomes = np.stack(all_scores, axis=1)  # (N_total, 10)
    else:
        outcomes, _, _ = classify_state(
            model,
            all_batched,
            n_classes=num_classes,
            Z0=False,
            class_labels=None,
            classify_type="classical",
        )

    # find accuracy
    predicted_labels = np.argmax(outcomes, axis=1)
    class_accuracies = []
    total_correct = 0
    total_samples = 0
    idx = 0
    for label_idx, n in zip(label_map, cluster_sizes):
        preds = predicted_labels[idx : idx + n]
        correct = int(np.sum(preds == label_idx))
        total_correct += correct
        total_samples += n
        class_accuracies.append(correct / n)
        idx += n

    overall_acc = total_correct / total_samples
    return class_accuracies, overall_acc


#  functions for plotting and truncating MPO at different bond dimensions,


def sweep_single_class(c, original_mpo_list, sweep_depth):
    """Sweep one class chain to canonical gauge"""
    t0 = original_mpo_list[0][c].reshape(1, 4, -1)
    current_chain = [t0] + [t.copy() for t in original_mpo_list[1:]]
    swept_chain, _, _ = left_to_right_sweep(current_chain, depth=sweep_depth)
    return c, swept_chain


def eval_class_vs_ensemble(label_idx, cluster, ensemble):
    """
    Score one class cluster against the full ensemble
    Returns (label_idx, per_class_acc, n_correct, n_total)
    """
    batched = batch_mps_cluster(cluster)
    chain_scores = []
    for chain in ensemble:
        s, _, _ = classify_state(
            chain,
            batched,
            n_classes=1,
            Z0=True,
            class_labels=None,
            classify_type="classical",
        )
        chain_scores.append(s.flatten())
    outcomes = np.stack(chain_scores, axis=1)  # (N, 10)
    preds = np.argmax(outcomes, axis=1)
    correct = int(np.sum(preds == label_idx))
    return label_idx, correct / len(cluster), correct, len(cluster)


def construct_ensemble(
    original_mpo_list,
    modifications_dict,
    sweep_depth=3,
    truncate_n=None,
):
    """
    Build 10 per-class MPS chains with optional motif swaps and bond truncation.

    Args
    ---
    original_mpo_list :
        trained MPO tensor list
    modifications_dict: dict
        {target_class: src_class} — replaces the full motif sub-chain of target_class
        with the sub-chain from src_class starting at sweep_depth
    sweep_depth: int
        site index where modification is applied (default 3
    truncate_n : int or None
        if set, truncates bond dim at sweep_depth after swapping

    Returns
    ---
    ensemble:list
        list of 10 MPS chains, ensemble[c] for class c
    """
    raw = Parallel(n_jobs=-1)(
        delayed(sweep_single_class)(c, original_mpo_list, sweep_depth)
        for c in range(10)
    )
    raw.sort(key=lambda x: x[0])
    base_swept_chains = [chain for _, chain in raw]

    final_ensemble = []

    for class_idx in range(10):
        current_chain = [t.copy() for t in base_swept_chains[class_idx]]

        # apply full motif sub-chain swap
        if class_idx in modifications_dict:
            src_class = modifications_dict[class_idx]
            current_chain[sweep_depth:] = [
                t.copy() for t in base_swept_chains[src_class][sweep_depth:]
            ]

        # apply truncation
        if truncate_n is not None:
            T_left = current_chain[sweep_depth - 1]
            T_right = current_chain[sweep_depth]

            # truncate last dim of left tensor, first dim of right
            T_left = T_left[:, :, :truncate_n]
            T_right = T_right[:truncate_n, :, :]

            current_chain[sweep_depth - 1] = T_left
            current_chain[sweep_depth] = T_right

        final_ensemble.append(current_chain)

    return final_ensemble


def plot_ensemble_performance(
    mpo_tensor_list,
    mods,
    cluster_list_test,
    baseline_stats,
    sweep_depth=3,
    truncate_n=None,
    title_str="",
):
    """
    Plot class-wise accuracy: original MPO vs modified ensemble.

    Args
    ---
    mpo_tensor_list :
        trained mpo
    mods:dict
        {target_class: src_class}
    cluster_list_test: list of lists
        MPS test states per class
    baseline_stats:tuple
        precomputed (class_accs, overall_acc), accuracy of original mpo
    sweep_depth: int
        modification site index (default 3)
    truncate_n: int, list, or None
        bond truncation value
    title_str
        appended to plot title

    Returns
    ---
    fig :
        matplotlib figure
    results_dict: dict
        truncate_n -> (class_accuracies, overall_accuracy)
    final_ensemble:list
        ensemble from the last truncation value
    """

    accuracies_original, acc_original = baseline_stats

    # convert truncate_n to list
    if truncate_n is None:
        trunc_vals = [None]
    elif isinstance(truncate_n, (list, tuple, np.ndarray)):
        trunc_vals = truncate_n
    else:
        trunc_vals = [truncate_n]

    results_dict = {}

    for t_n in trunc_vals:
        print(f"Evaluating ensemble with truncate_n={t_n}")

        final_ensemble = construct_ensemble(
            mpo_tensor_list,
            mods,
            sweep_depth=sweep_depth,
            truncate_n=t_n,
        )

        raw_eval = Parallel(n_jobs=-1, prefer="threads")(
            delayed(eval_class_vs_ensemble)(cls, cluster_list_test[cls], final_ensemble)
            for cls in range(10)
            if len(cluster_list_test[cls]) > 0
        )
        raw_eval.sort(key=lambda x: x[0])
        class_acc_current = [acc for _, acc, _, _ in raw_eval]
        total_correct = sum(c for _, _, c, _ in raw_eval)
        total_samples = sum(n for _, _, _, n in raw_eval)
        overall_acc = total_correct / total_samples

        results_dict[t_n] = (class_acc_current, overall_acc)

    fig = plt.figure(figsize=(12, 6))

    # plot baseline
    plt.plot(
        accuracies_original,
        color="black",
        linestyle="--",
        marker="s",
        linewidth=2,
        label=f"Original MPO (Acc={acc_original:.2%})",
    )
    markers = ["o", "^", "D", "v", "x", "*", "P", "H", "<", ">"]
    for i, (t_n, (accs, overall)) in enumerate(results_dict.items()):
        marker = markers[i % len(markers)]
        label_str = f"Best {t_n} motif" if t_n is not None else "Mod (No Trunc)"

        plt.plot(
            accs, marker=marker, alpha=0.8, label=f"{label_str} (Acc={overall:.2%})"
        )

    plt.xticks(range(10))
    plt.ylim(0.3, 1.05)
    plt.xlabel("Class Label")
    plt.ylabel("Accuracy")
    if title_str is not None:
        plt.title(f"Class-wise Accuracy Comparison\n {title_str}")
    else:
        plt.title("Class-wise Accuracy Comparison\n(Baseline vs. Modified Ensembles)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # info table — modifications list
    header = "source class -> target class (full motif sub-chain swap)"
    lines = [header, "-" * len(header)]

    if mods is None or len(mods) == 0:
        pass
    else:
        for target_class, src_class in sorted(mods.items()):
            lines.append(f"class {src_class} -> class {target_class}")

    if mods is not None and len(mods) > 0:
        plt.text(
            0.02,
            0.05,
            "\n".join(lines),
            transform=plt.gca().transAxes,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9),
        )

    return fig, results_dict, final_ensemble


def truncate_mpo_via_gauge(chain, bond_dims):
    """
    Truncate MPS bond dims via SVD at each bond.

    Args
    ---
    chain:list of ndarray
        rank-3 tensors (left, phys, right)
    bond_dims:list of int
        target chi per internal bond; for N sites give N-1 values

    Returns
    ---
    new_chain: list of ndarray
        truncated MPS tensor
    singular_values: list of ndarray
        kept singular values per bond
    """
    num_sites = len(chain)

    new_chain = [t.copy() for t in chain]
    singular_values = []

    for i in range(num_sites - 1):
        A = new_chain[i]
        left, phys, right = A.shape

        # reshape to matrix: (left * phys, right)
        M = A.reshape(left * phys, right)

        U, s, Vd = svd(M, full_matrices=False)

        # truncate to target bond dim
        chi = min(len(s), bond_dims[i])
        U = U[:, :chi]
        s_trunc = s[:chi]
        Vd = Vd[:chi, :]

        singular_values.append(s_trunc)

        # reshape U back to tensor (left, phys, chi)
        new_chain[i] = U.reshape(left, phys, chi)

        # absorb S * Vd into next tensor
        sVd = np.diag(s_trunc) @ Vd  # (chi, right)
        A_next = new_chain[i + 1]  # (right, phys_next, right_next)
        new_chain[i + 1] = np.tensordot(sVd, A_next, axes=([1], [0]))

    return new_chain, singular_values


# --------
#  functions for finding the optimal bond configuration for an mpo
# --------


def count_params(bond_config):
    """Count total parameters in a 5-site MPS given bond config"""
    b = bond_config
    return b[0] + b[0] * b[1] + b[1] * b[2] + b[2] * b[3] + b[3]


def eval_config_full_mpo(config, mpo_list, test_clusters):
    """Truncate the MPO and evaluate with ensemble_mode=False."""
    compressed, _ = truncate_mpo_via_gauge(list(mpo_list), list(config))
    _, acc = evaluate_accuracy(compressed, test_clusters, ensemble_mode=False)
    return config, acc


def grid_search_mpo_config(mpo_list, test_clusters, bounds, accuracy_threshold):
    """
    Parallel grid search over per bond dimensions to find the config with the
    fewest parameters that meets accuracy_threshold.

    Args
    ---
    mpo_list :
        full MPO tensor list
    test_clusters :
        test clusters per class
    bounds:list of (int, int)
        [(lo, hi), ...] for each internal bond
    accuracy_threshold:float
        min acceptable accuracy

    Returns
    ---
    best_config: ints
        config with fewest parameters meeting the threshold.
    best_acc: float
        accuracy at best_config
    all_results: list of (config, acc)
        all evaluated configs sorted by parameter count
    """
    configs = list(itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]))

    raw = Parallel(n_jobs=-1)(
        delayed(eval_config_full_mpo)(cfg, mpo_list, test_clusters) for cfg in configs
    )

    passing = [(cfg, acc) for cfg, acc in raw if acc >= accuracy_threshold]
    passing.sort(key=lambda x: count_params(x[0]))

    all_results = sorted(raw, key=lambda x: count_params(x[0]))

    if not passing:
        print("No config found")
    else:
        best_config, best_acc = passing[0]

    return best_config, best_acc, all_results
