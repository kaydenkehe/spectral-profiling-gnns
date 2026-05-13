import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

pd = None
torch = None
F = None
LogisticRegression = None
Ridge = None
accuracy_score = None
mean_absolute_error = None
mean_squared_error = None
r2_score = None
make_pipeline = None
StandardScaler = None
Parallel = None
delayed = None
get_laplacian = None
to_undirected = None


def load_runtime_dependencies():
    global pd, torch, F
    global LogisticRegression, Ridge, accuracy_score, mean_absolute_error
    global mean_squared_error, r2_score, make_pipeline, StandardScaler
    global Parallel, delayed, get_laplacian, to_undirected

    print("[startup] importing pandas", flush=True)
    import pandas as pd_module  # pylint: disable=import-outside-toplevel
    print("[startup] importing torch", flush=True)
    import torch as torch_module  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as functional_module  # pylint: disable=import-outside-toplevel
    print("[startup] importing sklearn", flush=True)
    from sklearn.linear_model import (  # pylint: disable=import-outside-toplevel
        LogisticRegression as LogisticRegressionClass,
        Ridge as RidgeClass,
    )
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        accuracy_score as accuracy_score_fn,
        mean_absolute_error as mean_absolute_error_fn,
        mean_squared_error as mean_squared_error_fn,
        r2_score as r2_score_fn,
    )
    from sklearn.pipeline import make_pipeline as make_pipeline_fn  # pylint: disable=import-outside-toplevel
    from sklearn.preprocessing import StandardScaler as StandardScalerClass  # pylint: disable=import-outside-toplevel
    print("[startup] importing joblib", flush=True)
    from joblib import Parallel as ParallelClass, delayed as delayed_fn  # pylint: disable=import-outside-toplevel
    print("[startup] importing torch_geometric.utils", flush=True)
    from torch_geometric.utils import (  # pylint: disable=import-outside-toplevel
        get_laplacian as get_laplacian_fn,
        to_undirected as to_undirected_fn,
    )
    print("[startup] imports complete", flush=True)

    pd = pd_module
    torch = torch_module
    F = functional_module
    LogisticRegression = LogisticRegressionClass
    Ridge = RidgeClass
    accuracy_score = accuracy_score_fn
    mean_absolute_error = mean_absolute_error_fn
    mean_squared_error = mean_squared_error_fn
    r2_score = r2_score_fn
    make_pipeline = make_pipeline_fn
    StandardScaler = StandardScalerClass
    Parallel = ParallelClass
    delayed = delayed_fn
    get_laplacian = get_laplacian_fn
    to_undirected = to_undirected_fn


def ensure_runtime_dependencies():
    if make_pipeline is None or Parallel is None:
        load_runtime_dependencies()


PRIMARY_DATASETS = [
    "Actor",
    "AmazonRatings",
    "Chameleon",
    "CiteSeer",
    "Computers",
    "Cora",
    "Cornell",
    "Minesweeper",
    "Photo",
    "PubMed",
    "Questions",
    "RomanEmpire",
    "Squirrel",
    "Texas",
    "Tolokers",
    "WikiCS",
    "Wisconsin",
]

FEATURE_FAMILIES = [
    "homophily",
    "metadata",
    "label_slp",
    "label_slp_homophily",
    "feature_energy",
    "feature_density",
    "label_slp_feature_density",
    "label_feature_product",
    "label_feature_gap",
    "band_cka",
    "combined",
]


def dataset_key(name):
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "amazonratings": "amazonratings",
        "amazonrating": "amazonratings",
        "amazonphoto": "photo",
        "phot": "photo",
        "amazoncomputers": "computers",
        "coauthorcs": "cs",
        "romanempire": "romanempire",
        "wikics": "wikics",
    }
    return aliases.get(key, key)


def canonical_dataset_name(name):
    key = dataset_key(name)
    names = {dataset_key(n): n for n in PRIMARY_DATASETS}
    extras = {"cs": "CS", "physics": "Physics"}
    return names.get(key, extras.get(key, name))


def values_to_arg_string(values):
    return ",".join(str(v) for v in values)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return repo_root() / path


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    if not fieldnames:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CsvStream:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def load_dataset_map(requested_names):
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from torch_geometric.datasets import (  # pylint: disable=import-outside-toplevel
        Actor,
        Amazon,
        Coauthor,
        HeterophilousGraphDataset,
        Planetoid,
        WebKB,
        WikiCS,
    )
    import torch_geometric.transforms as T  # pylint: disable=import-outside-toplevel
    from common.datasets import FixedWikipediaNetwork  # pylint: disable=import-outside-toplevel

    data_dir = str(root / "graph_data")
    os.makedirs(data_dir, exist_ok=True)

    builders = {
        "Actor": lambda: Actor(root=os.path.join(data_dir, "actor")),
        "AmazonRatings": lambda: HeterophilousGraphDataset(root=data_dir, name="Amazon-ratings"),
        "Chameleon": lambda: FixedWikipediaNetwork(root=data_dir, name="Chameleon"),
        "CiteSeer": lambda: Planetoid(root=data_dir, name="CiteSeer"),
        "Computers": lambda: Amazon(root=data_dir, name="Computers"),
        "Cora": lambda: Planetoid(root=data_dir, name="Cora"),
        "Cornell": lambda: WebKB(root=data_dir, name="Cornell"),
        "CS": lambda: Coauthor(root=data_dir, name="CS"),
        "Minesweeper": lambda: HeterophilousGraphDataset(
            root=data_dir,
            name="Minesweeper",
            pre_transform=T.ToUndirected(),
        ),
        "Photo": lambda: Amazon(root=data_dir, name="Photo"),
        "Physics": lambda: Coauthor(root=data_dir, name="Physics"),
        "PubMed": lambda: Planetoid(root=data_dir, name="PubMed"),
        "Questions": lambda: HeterophilousGraphDataset(
            root=data_dir,
            name="Questions",
            pre_transform=T.ToUndirected(),
        ),
        "RomanEmpire": lambda: HeterophilousGraphDataset(root=data_dir, name="Roman-empire"),
        "Squirrel": lambda: FixedWikipediaNetwork(root=data_dir, name="Squirrel"),
        "Texas": lambda: WebKB(root=data_dir, name="Texas"),
        "Tolokers": lambda: HeterophilousGraphDataset(
            root=data_dir,
            name="Tolokers",
            pre_transform=T.ToUndirected(),
        ),
        "WikiCS": lambda: WikiCS(root=os.path.join(data_dir, "wikics")),
        "Wisconsin": lambda: WebKB(root=data_dir, name="Wisconsin"),
    }

    dataset_map = {}
    for name in requested_names:
        canonical_name = canonical_dataset_name(name)
        if canonical_name in builders:
            dataset_map[canonical_name] = builders[canonical_name]()
    return dataset_map


def to_dense_feature_matrix(x):
    if x is None:
        raise ValueError("dataset has no node features")
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    return x.float()


def compute_homophily(edge_index, labels, node_mask=None, num_classes=None):
    if node_mask is not None:
        src, dst = edge_index
        edge_index = edge_index[:, node_mask[src] & node_mask[dst]]
    if edge_index.numel() == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    src, dst = edge_index
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).float()
    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()


def normalized_laplacian_dense(edge_index, num_nodes, dtype):
    edge_index = to_undirected(edge_index.cpu(), num_nodes=num_nodes)
    edge_index, edge_weight = get_laplacian(
        edge_index,
        normalization="sym",
        num_nodes=num_nodes,
    )
    lap = torch.sparse_coo_tensor(
        edge_index,
        edge_weight.to(dtype=dtype),
        (num_nodes, num_nodes),
    ).coalesce()
    return lap.to_dense()


def normalized_laplacian_sparse(edge_index, num_nodes, dtype, device):
    edge_index = to_undirected(edge_index.cpu(), num_nodes=num_nodes)
    edge_index, edge_weight = get_laplacian(
        edge_index,
        normalization="sym",
        num_nodes=num_nodes,
    )
    lap = torch.sparse_coo_tensor(
        edge_index.to(device),
        edge_weight.to(device=device, dtype=dtype),
        (num_nodes, num_nodes),
    ).coalesce()
    return lap


def centered_one_hot(labels, num_classes):
    y = F.one_hot(labels, num_classes=num_classes).float()
    return y - y.mean(dim=0, keepdim=True)


def train_centered_one_hot(labels, num_classes, train_mask):
    y = F.one_hot(labels, num_classes=num_classes).float()
    out = torch.zeros_like(y)
    if train_mask.any():
        out[train_mask] = y[train_mask] - y[train_mask].mean(dim=0, keepdim=True)
    return out


def random_train_mask(num_nodes, seed, train_r=0.6):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=generator)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[perm[:int(train_r * num_nodes)]] = True
    return mask


def native_or_random_train_mask(data, seed):
    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_mask = data.train_mask
        if train_mask.dim() == 2:
            return train_mask[:, int(seed) % train_mask.size(1)].cpu().bool()
        return train_mask.cpu().bool()
    return random_train_mask(int(data.num_nodes), seed)


def train_masks_for_label_scope(data, seeds, split_mode):
    if split_mode == "paper_random":
        return [random_train_mask(int(data.num_nodes), seed) for seed in seeds]
    if split_mode == "jacobi_native_or_random":
        return [native_or_random_train_mask(data, seed) for seed in seeds]
    raise ValueError(f"unknown label split mode: {split_mode}")


def label_signals(labels, num_classes, label_scope, train_masks, device, dtype):
    if label_scope == "full":
        return [centered_one_hot(labels, num_classes).to(device=device, dtype=dtype)]
    return [
        train_centered_one_hot(labels, num_classes, mask).to(device=device, dtype=dtype)
        for mask in train_masks
    ]


def normalized_label_energy(evecs, y_centered):
    proj = evecs.T @ y_centered
    denom = torch.sum(y_centered ** 2).clamp(min=1e-12)
    return (torch.sum(proj ** 2, dim=1) / denom).cpu().numpy()


def feature_energy(evecs, x_centered):
    proj = evecs.T @ x_centered
    denom = torch.sum(x_centered ** 2).clamp(min=1e-12)
    return (torch.sum(proj ** 2, dim=1) / denom).cpu().numpy(), proj


def label_projection(evecs, y_centered):
    return evecs.T @ y_centered


def linear_cka_from_projected(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    numerator = torch.sum((a.T @ b) ** 2)
    aa = torch.linalg.matrix_norm(a.T @ a, ord="fro")
    bb = torch.linalg.matrix_norm(b.T @ b, ord="fro")
    denom = aa * bb
    if denom.item() <= 1e-12:
        return 0.0
    return float((numerator / denom).item())


def bin_edges(n_bins):
    return np.linspace(0.0, 2.0, n_bins + 1)


def aggregate_by_bins(evals, values, n_bins):
    edges = bin_edges(n_bins)
    out = []
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (evals >= left) & (evals <= right)
        else:
            mask = (evals >= left) & (evals < right)
        out.append(float(values[mask].sum()) if np.any(mask) else 0.0)
    return np.array(out, dtype=float)


def jackson_damping(order):
    if order <= 0:
        return np.ones(1, dtype=float)
    k = np.arange(order + 1, dtype=float)
    theta = np.pi / (order + 1)
    return (
        (order - k + 1) * np.cos(k * theta)
        + np.sin(k * theta) / np.tan(theta)
    ) / (order + 1)


def chebyshev_interval_coefficients(left_lambda, right_lambda, order, use_jackson):
    lo = float(np.clip(left_lambda - 1.0, -1.0, 1.0))
    hi = float(np.clip(right_lambda - 1.0, -1.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo

    theta_left = math.acos(hi)
    theta_right = math.acos(lo)
    coeffs = np.zeros(order + 1, dtype=float)
    coeffs[0] = (theta_right - theta_left) / math.pi
    for k in range(1, order + 1):
        coeffs[k] = (
            2.0
            / math.pi
            * (math.sin(k * theta_right) - math.sin(k * theta_left))
            / k
        )
    if use_jackson:
        coeffs *= jackson_damping(order)
    return coeffs


def shifted_laplacian_mm(lap, signal):
    return torch.sparse.mm(lap, signal) - signal


def chebyshev_filter_signal(lap, signal, coeffs):
    coeffs = torch.as_tensor(coeffs, device=signal.device, dtype=signal.dtype)
    out = coeffs[0] * signal
    if coeffs.numel() == 1:
        return out

    t_prev = signal
    t_curr = shifted_laplacian_mm(lap, signal)
    out = out + coeffs[1] * t_curr
    for idx in range(2, coeffs.numel()):
        t_next = 2.0 * shifted_laplacian_mm(lap, t_curr) - t_prev
        out = out + coeffs[idx] * t_next
        t_prev, t_curr = t_curr, t_next
    return out


def normalized_overlap(original, projected, denom):
    value = torch.sum(original * projected) / denom.clamp(min=1e-12)
    return float(torch.clamp(value, min=0.0, max=1.0).item())


def feature_component_name(feature_family, feature_index, n_bins):
    if feature_family == "homophily":
        return "homophily", -1
    if feature_family == "metadata":
        return ["log_num_nodes", "log_num_edges", "log_num_features", "log_num_classes"][feature_index], -1
    if feature_family == "label_slp":
        return "label_slp_mass", feature_index
    if feature_family == "label_slp_homophily":
        if feature_index < n_bins:
            return "label_slp_mass", feature_index
        return "homophily", -1
    if feature_family == "feature_energy":
        return "feature_energy_mass", feature_index
    if feature_family == "feature_density":
        return "feature_spectral_density", feature_index
    if feature_family == "label_slp_feature_density":
        if feature_index < n_bins:
            return "label_slp_mass", feature_index
        return "feature_spectral_density", feature_index - n_bins
    if feature_family == "label_feature_product":
        return "label_feature_product", feature_index
    if feature_family == "label_feature_gap":
        return "label_feature_gap", feature_index
    if feature_family == "band_cka":
        return "band_cka", feature_index
    if feature_family == "combined":
        spans = [
            ("label_slp_mass", n_bins),
            ("feature_energy_mass", n_bins),
            ("feature_spectral_density", n_bins),
            ("label_feature_product", n_bins),
            ("label_feature_gap", n_bins),
            ("band_cka", n_bins),
            ("homophily", 1),
        ]
        offset = 0
        for component, width in spans:
            if feature_index < offset + width:
                if component == "homophily":
                    return component, -1
                return component, feature_index - offset
            offset += width
    return f"{feature_family}_{feature_index}", -1


def compute_exact_profiles(edge_index, num_nodes, x_centered, y_centered_list, bins, dtype, device):
    lap = normalized_laplacian_dense(edge_index, num_nodes, dtype).to(device)
    evals, evecs = torch.linalg.eigh(lap)

    feat_energy, projected_x = feature_energy(evecs, x_centered)
    evals_np = evals.cpu().numpy()

    profiles = {}
    for n_bins in bins:
        feature_mass = aggregate_by_bins(evals_np, feat_energy, n_bins)
        label_masses = []
        ckas = []
        edges = bin_edges(n_bins)
        bin_masks = []
        for bin_idx in range(n_bins):
            left, right = edges[bin_idx], edges[bin_idx + 1]
            if bin_idx == n_bins - 1:
                mask_np = (evals_np >= left) & (evals_np <= right)
            else:
                mask_np = (evals_np >= left) & (evals_np < right)
            bin_masks.append(torch.as_tensor(mask_np, device=device, dtype=torch.bool))
        for y_centered in y_centered_list:
            label_energy = normalized_label_energy(evecs, y_centered)
            projected_y = label_projection(evecs, y_centered)
            label_masses.append(aggregate_by_bins(evals_np, label_energy, n_bins))
            ckas.append(np.asarray([
                linear_cka_from_projected(projected_x[mask], projected_y[mask])
                for mask in bin_masks
            ], dtype=float))
        profiles[n_bins] = {
            "label_mass": np.mean(label_masses, axis=0),
            "feature_mass": feature_mass,
            "band_cka": np.mean(ckas, axis=0),
        }

    del lap, evals, evecs, projected_x
    return profiles


def compute_chebyshev_profiles(edge_index, num_nodes, x_centered, y_centered_list, bins,
                               dtype, device, order, use_jackson):
    lap = normalized_laplacian_sparse(edge_index, num_nodes, dtype, device)
    signal = torch.cat([x_centered, *y_centered_list], dim=1)
    x_width = x_centered.size(1)
    y_width = y_centered_list[0].size(1)
    x_denom = torch.sum(x_centered ** 2)
    y_denoms = [torch.sum(y_centered ** 2) for y_centered in y_centered_list]

    profiles = {}
    for n_bins in bins:
        edges = bin_edges(n_bins)
        label_mass = []
        feature_mass = []
        cka = []
        for bin_idx in range(n_bins):
            coeffs = chebyshev_interval_coefficients(
                edges[bin_idx],
                edges[bin_idx + 1],
                order,
                use_jackson,
            )
            projected = chebyshev_filter_signal(lap, signal, coeffs)
            projected_x = projected[:, :x_width]
            feature_mass.append(normalized_overlap(x_centered, projected_x, x_denom))
            label_bin_mass = []
            label_bin_cka = []
            for idx, y_centered in enumerate(y_centered_list):
                start = x_width + idx * y_width
                projected_y = projected[:, start:start + y_width]
                label_bin_mass.append(normalized_overlap(y_centered, projected_y, y_denoms[idx]))
                label_bin_cka.append(linear_cka_from_projected(projected_x, projected_y))
            label_mass.append(float(np.mean(label_bin_mass)))
            cka.append(float(np.mean(label_bin_cka)))
            del projected, projected_x

        profiles[n_bins] = {
            "label_mass": np.asarray(label_mass, dtype=float),
            "feature_mass": np.asarray(feature_mass, dtype=float),
            "band_cka": np.asarray(cka, dtype=float),
        }

    del lap, signal
    return profiles


def lanczos_measure(lap, signal, steps):
    norm = torch.linalg.norm(signal)
    if norm.item() <= 1e-12:
        return np.array([], dtype=float), np.array([], dtype=float)

    q = signal / norm
    q_prev = torch.zeros_like(q)
    beta_prev = signal.new_tensor(0.0)
    basis = []
    alphas = []
    betas = []

    for _ in range(steps):
        basis.append(q)
        z = torch.sparse.mm(lap, q[:, None]).squeeze(1)
        if len(basis) > 1:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        z = z - alpha * q
        for old_q in basis:
            z = z - torch.dot(old_q, z) * old_q
        beta = torch.linalg.norm(z)
        alphas.append(alpha)
        if len(alphas) == steps or beta.item() <= 1e-10:
            break
        betas.append(beta)
        q_prev, q, beta_prev = q, z / beta, beta

    t = torch.diag(torch.stack(alphas))
    if betas:
        off_diag = torch.stack(betas)
        t = t + torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)
    evals, evecs = torch.linalg.eigh(t)
    weights = (evecs[0] ** 2) * (norm ** 2)
    return (
        np.clip(evals.detach().cpu().numpy(), 0.0, 2.0),
        weights.detach().cpu().numpy(),
    )


def lanczos_bin_masses(lap, signals, bins, steps, denom, scale=1.0):
    masses = {n_bins: np.zeros(n_bins, dtype=float) for n_bins in bins}
    if denom.item() <= 1e-12:
        return masses

    for col_idx in range(signals.size(1)):
        evals, weights = lanczos_measure(lap, signals[:, col_idx], steps)
        if evals.size == 0:
            continue
        for n_bins in bins:
            masses[n_bins] += aggregate_by_bins(evals, weights, n_bins)
    return {
        n_bins: scale * mass / float(denom.item())
        for n_bins, mass in masses.items()
    }


def rademacher_feature_probes(x_centered, num_probes, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (x_centered.size(1), num_probes),
        generator=generator,
        dtype=torch.float32,
    ).mul_(2).sub_(1)
    signs = signs.to(device=x_centered.device, dtype=x_centered.dtype)
    return x_centered @ signs


def compute_lanczos_profiles(edge_index, num_nodes, x_centered, y_centered_list, bins,
                             dtype, device, steps, feature_probes, seed):
    lap = normalized_laplacian_sparse(edge_index, num_nodes, dtype, device)
    x_denom = torch.sum(x_centered ** 2)
    probes = rademacher_feature_probes(x_centered, feature_probes, seed)
    feature_masses = lanczos_bin_masses(
        lap,
        probes,
        bins,
        steps,
        x_denom,
        scale=1.0 / feature_probes,
    )

    label_by_seed = []
    for y_centered in y_centered_list:
        label_by_seed.append(lanczos_bin_masses(
            lap,
            y_centered,
            bins,
            steps,
            torch.sum(y_centered ** 2),
        ))

    profiles = {}
    for n_bins in bins:
        profiles[n_bins] = {
            "label_mass": np.mean([masses[n_bins] for masses in label_by_seed], axis=0),
            "feature_mass": feature_masses[n_bins],
            "band_cka": np.zeros(n_bins, dtype=float),
        }
    del lap, probes
    return profiles


def select_spectral_method(method, num_nodes, max_dense_elements):
    dense_elements = num_nodes * num_nodes
    if method == "auto":
        if max_dense_elements == 0 or dense_elements <= max_dense_elements:
            return "exact"
        return "chebyshev"
    return method


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU", flush=True)
        return torch.device("cpu")
    if device.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            print("[device] MPS requested but unavailable; using CPU", flush=True)
            return torch.device("cpu")
    return device


def metric_compute_device(requested_device, method_used):
    if requested_device.type == "mps" and method_used in {"chebyshev", "lanczos"}:
        return torch.device("cpu"), f"mps_sparse_mm_unsupported_for_{method_used}"
    return requested_device, ""


def compute_dataset_metrics(dataset_name, dataset, bins, dtype, device, max_dense_elements,
                            spectral_method, chebyshev_order, chebyshev_jackson,
                            label_scope, label_seeds, label_split_mode,
                            lanczos_steps, lanczos_feature_probes, lanczos_seed,
                            feature_families):
    data = dataset[0]
    num_nodes = int(data.num_nodes)
    method_used = select_spectral_method(spectral_method, num_nodes, max_dense_elements)
    compute_device, fallback_reason = metric_compute_device(device, method_used)
    if (
        method_used == "exact"
        and max_dense_elements
        and num_nodes * num_nodes > max_dense_elements
        and spectral_method == "exact"
    ):
        raise RuntimeError(
            f"exact dense eigendecomposition skipped: {num_nodes}^2 exceeds "
            f"--max-dense-elements={max_dense_elements}"
        )

    graph = data.cpu()
    x = to_dense_feature_matrix(graph.x).to(device=compute_device, dtype=dtype)
    edge_index = graph.edge_index.to(compute_device)

    x_centered = x - x.mean(dim=0, keepdim=True)
    train_masks = (
        train_masks_for_label_scope(graph, label_seeds, label_split_mode)
        if label_scope == "train"
        else []
    )
    y_centered_list = label_signals(
        graph.y,
        dataset.num_classes,
        label_scope,
        train_masks,
        compute_device,
        dtype,
    )

    if label_scope == "train":
        homophily = float(np.mean([
            compute_homophily(
                graph.edge_index,
                graph.y,
                node_mask=mask,
                num_classes=int(dataset.num_classes),
            )
            for mask in train_masks
        ]))
    else:
        homophily = compute_homophily(
            graph.edge_index,
            graph.y,
            num_classes=int(dataset.num_classes),
        )
    metadata = {
        "num_nodes": num_nodes,
        "num_edges": int(graph.edge_index.size(1)),
        "num_features": int(x.size(1)),
        "num_classes": int(dataset.num_classes),
        "homophily": float(homophily),
        "label_scope": label_scope,
        "label_split_mode": label_split_mode if label_scope == "train" else "",
        "label_seeds": values_to_arg_string(label_seeds) if label_scope == "train" else "",
        "label_seed_count": len(label_seeds) if label_scope == "train" else 0,
        "spectral_method": method_used,
        "requested_device": device.type,
        "compute_device": compute_device.type,
        "device_fallback_reason": fallback_reason,
        "chebyshev_order": int(chebyshev_order) if method_used == "chebyshev" else "",
        "chebyshev_jackson": bool(chebyshev_jackson) if method_used == "chebyshev" else "",
        "lanczos_steps": int(lanczos_steps) if method_used == "lanczos" else "",
        "lanczos_feature_probes": int(lanczos_feature_probes) if method_used == "lanczos" else "",
        "lanczos_seed": int(lanczos_seed) if method_used == "lanczos" else "",
    }

    if method_used == "exact":
        try:
            profiles = compute_exact_profiles(
                edge_index,
                num_nodes,
                x_centered,
                y_centered_list,
                bins,
                dtype,
                compute_device,
            )
        except (RuntimeError, NotImplementedError) as exc:
            if compute_device.type != "mps":
                raise
            compute_device = torch.device("cpu")
            metadata["compute_device"] = "cpu"
            metadata["device_fallback_reason"] = f"mps_exact_failed: {exc}"
            x_centered = x_centered.cpu()
            y_centered_list = [y_centered.cpu() for y_centered in y_centered_list]
            edge_index = edge_index.cpu()
            profiles = compute_exact_profiles(
                edge_index,
                num_nodes,
                x_centered,
                y_centered_list,
                bins,
                dtype,
                compute_device,
            )
    elif method_used == "chebyshev":
        profiles = compute_chebyshev_profiles(
            edge_index,
            num_nodes,
            x_centered,
            y_centered_list,
            bins,
            dtype,
            compute_device,
            chebyshev_order,
            chebyshev_jackson,
        )
    elif method_used == "lanczos":
        profiles = compute_lanczos_profiles(
            edge_index,
            num_nodes,
            x_centered,
            y_centered_list,
            bins,
            dtype,
            compute_device,
            lanczos_steps,
            lanczos_feature_probes,
            lanczos_seed,
        )
    else:
        raise ValueError(f"unknown spectral method: {spectral_method}")

    metric_rows = []
    feature_vectors = {}
    for n_bins in bins:
        label_mass = profiles[n_bins]["label_mass"]
        feature_mass = profiles[n_bins]["feature_mass"]
        edges = bin_edges(n_bins)
        bin_widths = np.diff(edges)
        feature_density = feature_mass / np.maximum(bin_widths, 1e-12)
        product = label_mass * feature_mass
        gap = np.abs(label_mass - feature_mass)
        cka = profiles[n_bins]["band_cka"]

        meta_vec = np.array([
            math.log1p(metadata["num_nodes"]),
            math.log1p(metadata["num_edges"]),
            math.log1p(metadata["num_features"]),
            math.log1p(metadata["num_classes"]),
        ], dtype=float)

        feature_vectors[(dataset_name, n_bins, "homophily")] = np.array([homophily], dtype=float)
        feature_vectors[(dataset_name, n_bins, "metadata")] = meta_vec
        feature_vectors[(dataset_name, n_bins, "label_slp")] = label_mass
        feature_vectors[(dataset_name, n_bins, "label_slp_homophily")] = np.r_[label_mass, homophily]
        feature_vectors[(dataset_name, n_bins, "feature_energy")] = feature_mass
        feature_vectors[(dataset_name, n_bins, "feature_density")] = feature_density
        feature_vectors[(dataset_name, n_bins, "label_slp_feature_density")] = np.r_[
            label_mass,
            feature_density,
        ]
        feature_vectors[(dataset_name, n_bins, "label_feature_product")] = product
        feature_vectors[(dataset_name, n_bins, "label_feature_gap")] = gap
        feature_vectors[(dataset_name, n_bins, "band_cka")] = cka
        feature_vectors[(dataset_name, n_bins, "combined")] = np.r_[
            label_mass,
            feature_mass,
            feature_density,
            product,
            gap,
            cka,
            homophily,
        ]

        for feature_family in feature_families:
            values = feature_vectors[(dataset_name, n_bins, feature_family)]
            for feature_index, metric_value in enumerate(values):
                component, bin_idx = feature_component_name(
                    feature_family,
                    feature_index,
                    n_bins,
                )
                if 0 <= bin_idx < n_bins:
                    bin_left = edges[bin_idx]
                    bin_right = edges[bin_idx + 1]
                else:
                    bin_left = np.nan
                    bin_right = np.nan
                metric_rows.append({
                    "dataset": dataset_name,
                    "n_bins": n_bins,
                    "metric_family": feature_family,
                    "feature_index": feature_index,
                    "component": component,
                    "bin_index": bin_idx,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "metric_value": float(metric_value),
                    **metadata,
                })

    del profiles, x_centered, y_centered_list
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()
    elif compute_device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return metric_rows, feature_vectors, metadata


def load_paper_targets(paper_root, expected_runs):
    paths = sorted(Path(paper_root).glob("**/summary.csv"))
    if not paths:
        return {}, pd.DataFrame()
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["_source"] = str(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in ["K", "hidden", "lr", "weight_decay", "epochs", "patience", "seed", "test_acc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    config_cols = ["dataset", "model", "K", "hidden", "lr", "weight_decay", "epochs", "patience"]
    agg = (
        df.groupby(config_cols)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            n_runs=("test_acc", "size"),
            source_files=("_source", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
    )
    complete = agg[agg["n_runs"] == expected_runs].copy()
    targets = {}
    for dataset_name, group in complete.groupby("dataset"):
        row = group.sort_values("mean_test_acc", ascending=False).iloc[0]
        raw_models = sorted(df[df["dataset"] == dataset_name]["model"].dropna().unique())
        complete_models = sorted(group["model"].dropna().unique())
        targets[canonical_dataset_name(dataset_name)] = {
            "paper_model": row["model"],
            "paper_K": int(row["K"]),
            "paper_hidden": int(row["hidden"]),
            "paper_lr": float(row["lr"]),
            "paper_weight_decay": float(row["weight_decay"]),
            "paper_mean_test_acc": float(row["mean_test_acc"]),
            "paper_std_test_acc": float(row["std_test_acc"]) if pd.notna(row["std_test_acc"]) else np.nan,
            "paper_n_runs": int(row["n_runs"]),
            "paper_source_files": row["source_files"],
            "paper_models_available": values_to_arg_string(complete_models),
            "paper_raw_models_available": values_to_arg_string(raw_models),
        }
    return targets, complete


def load_jacobi_targets(jacobi_root):
    paths = sorted(Path(jacobi_root).glob("**/summary.csv"))
    if not paths:
        return {}, pd.DataFrame()
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["_source"] = str(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in ["K", "a", "b", "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc", "n_seeds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    targets = {}
    for dataset_name, group in df.groupby("dataset"):
        row = group.sort_values("mean_val_acc", ascending=False).iloc[0]
        targets[canonical_dataset_name(dataset_name)] = {
            "jacobi_a": float(row["a"]),
            "jacobi_b": float(row["b"]),
            "jacobi_K": int(row["K"]),
            "jacobi_mean_val_acc": float(row["mean_val_acc"]),
            "jacobi_mean_test_acc": float(row["mean_test_acc"]),
            "jacobi_std_test_acc": float(row["std_test_acc"]) if pd.notna(row["std_test_acc"]) else np.nan,
            "jacobi_n_seeds": int(row["n_seeds"]),
            "jacobi_source_files": ";".join(sorted(set(group["_source"]))),
            "jacobi_K_values_available": values_to_arg_string(sorted(group["K"].dropna().unique().astype(int))),
        }
    return targets, df


def build_target_table(datasets, paper_targets, jacobi_targets):
    rows = []
    for dataset_name in datasets:
        row = {"dataset": dataset_name}
        row.update(paper_targets.get(dataset_name, {}))
        row.update(jacobi_targets.get(dataset_name, {}))
        row["has_paper_target"] = dataset_name in paper_targets
        row["has_jacobi_target"] = dataset_name in jacobi_targets
        row["paper_target_selection_rule"] = (
            "best mean_test_acc among complete expected-run paper-massive configs"
            if row["has_paper_target"]
            else ""
        )
        row["architecture_target_selection_rule"] = (
            "paper_model from the selected paper-massive config"
            if row["has_paper_target"]
            else ""
        )
        row["k_target_selection_rule"] = (
            "paper_K from the selected paper-massive config"
            if row["has_paper_target"]
            else ""
        )
        row["jacobi_target_selection_rule"] = (
            "best mean_val_acc among Jacobi a,b,K configs"
            if row["has_jacobi_target"]
            else ""
        )
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_rows(datasets, metric_status, target_table):
    rows = []
    for dataset_name in datasets:
        metric_ok = metric_status.get(dataset_name, {}).get("ok", False)
        metric_reason = metric_status.get(dataset_name, {}).get("reason", "")
        target_row = target_table[target_table["dataset"] == dataset_name]
        has_paper = bool(target_row["has_paper_target"].iloc[0]) if not target_row.empty else False
        has_jacobi = bool(target_row["has_jacobi_target"].iloc[0]) if not target_row.empty else False
        for task, has_target in [
            ("jacobi_ab", has_jacobi),
            ("architecture", has_paper),
            ("paper_K", has_paper),
        ]:
            included = metric_ok and has_target
            if included:
                reason = "included"
            elif not metric_ok:
                reason = f"missing_metrics: {metric_reason}"
            else:
                reason = "missing_target"
            rows.append({
                "dataset": dataset_name,
                "task": task,
                "has_metrics": metric_ok,
                "has_target": has_target,
                "included": included,
                "reason": reason,
            })
    return rows


def make_candidates(task, bins, regression_alphas, logistic_cs, feature_families):
    candidates = []
    for feature_family in feature_families:
        for n_bins in bins:
            if task == "jacobi_ab":
                for alpha in regression_alphas:
                    candidates.append({
                        "feature_family": feature_family,
                        "n_bins": n_bins,
                        "model_kind": "ridge",
                        "regularization": alpha,
                    })
            else:
                for c_value in logistic_cs:
                    candidates.append({
                        "feature_family": feature_family,
                        "n_bins": n_bins,
                        "model_kind": "logistic",
                        "regularization": c_value,
                    })
    return candidates


def feature_matrix(names, candidate, feature_vectors):
    rows = []
    for name in names:
        key = (name, candidate["n_bins"], candidate["feature_family"])
        rows.append(feature_vectors[key])
    return np.vstack(rows)


def regression_targets(target_table, names):
    indexed = target_table.set_index("dataset")
    return indexed.loc[names, ["jacobi_a", "jacobi_b"]].to_numpy(dtype=float)


def classification_targets(target_table, names, task):
    indexed = target_table.set_index("dataset")
    if task == "architecture":
        return indexed.loc[names, "paper_model"].astype(str).to_numpy()
    if task == "paper_K":
        return indexed.loc[names, "paper_K"].astype(int).astype(str).to_numpy()
    raise ValueError(task)


def mean_regression_baseline(y_train, y_eval):
    pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_eval), axis=0)
    return pred


def majority_baseline(y_train, y_eval):
    majority = Counter(y_train).most_common(1)[0][0]
    return np.array([majority] * len(y_eval), dtype=object)


def regression_metrics(y_true, y_pred):
    return {
        "mae_a": mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        "mae_b": mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
        "rmse_a": mean_squared_error(y_true[:, 0], y_pred[:, 0]) ** 0.5,
        "rmse_b": mean_squared_error(y_true[:, 1], y_pred[:, 1]) ** 0.5,
        "r2_a": safe_r2(y_true[:, 0], y_pred[:, 0]),
        "r2_b": safe_r2(y_true[:, 1], y_pred[:, 1]),
    }


def safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return np.nan
    return r2_score(y_true, y_pred)


def fit_predict_regression(x_train, y_train, x_eval, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


def fit_predict_classification(x_train, y_train, x_eval, c_value):
    if len(np.unique(y_train)) < 2:
        pred = majority_baseline(y_train, np.zeros(len(x_eval)))
        return pred, None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=c_value),
    )
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


def evaluate_candidate(task, candidate, train_names, eval_names, target_table, feature_vectors):
    ensure_runtime_dependencies()
    x_train = feature_matrix(train_names, candidate, feature_vectors)
    x_eval = feature_matrix(eval_names, candidate, feature_vectors)

    if task == "jacobi_ab":
        y_train = regression_targets(target_table, train_names)
        y_eval = regression_targets(target_table, eval_names)
        pred, _ = fit_predict_regression(
            x_train,
            y_train,
            x_eval,
            alpha=float(candidate["regularization"]),
        )
        base = mean_regression_baseline(y_train, y_eval)
        metrics = regression_metrics(y_eval, pred)
        base_metrics = regression_metrics(y_eval, base)
        score = -float((metrics["mae_a"] + metrics["mae_b"]) / 2.0)
        base_score = -float((base_metrics["mae_a"] + base_metrics["mae_b"]) / 2.0)
        return score, base_score, metrics, base_metrics, pred, base, y_eval

    y_train = classification_targets(target_table, train_names, task)
    y_eval = classification_targets(target_table, eval_names, task)
    pred, _ = fit_predict_classification(
        x_train,
        y_train,
        x_eval,
        c_value=float(candidate["regularization"]),
    )
    base = majority_baseline(y_train, y_eval)
    acc = accuracy_score(y_eval, pred)
    base_acc = accuracy_score(y_eval, base)
    metrics = {"accuracy": acc}
    base_metrics = {"accuracy": base_acc}
    return acc, base_acc, metrics, base_metrics, pred, base, y_eval


def candidate_row(protocol, task, fold_id, candidate_id, candidate, train_names, test_names,
                  val_names, train_score, test_score, baseline_test_score, selected):
    return {
        "protocol": protocol,
        "selection_split": "test" if protocol == "nested_2_test_2_val" else "train",
        "task": task,
        "fold_id": fold_id,
        "candidate_id": candidate_id,
        "feature_family": candidate["feature_family"],
        "n_bins": candidate["n_bins"],
        "model_kind": candidate["model_kind"],
        "regularization": candidate["regularization"],
        "train_datasets": values_to_arg_string(train_names),
        "test_datasets": values_to_arg_string(test_names),
        "val_datasets": values_to_arg_string(val_names),
        "train_score": train_score,
        "test_score": test_score,
        "baseline_test_score": baseline_test_score,
        "selected": selected,
    }


def selected_row(protocol, task, fold_id, candidate_id, candidate, train_names, test_names,
                 val_names, test_score, val_score, baseline_val_score, val_metrics, baseline_val_metrics):
    row = {
        "protocol": protocol,
        "selection_split": "test" if protocol == "nested_2_test_2_val" else "train",
        "task": task,
        "fold_id": fold_id,
        "candidate_id": candidate_id,
        "feature_family": candidate["feature_family"],
        "n_bins": candidate["n_bins"],
        "model_kind": candidate["model_kind"],
        "regularization": candidate["regularization"],
        "train_datasets": values_to_arg_string(train_names),
        "test_datasets": values_to_arg_string(test_names),
        "val_datasets": values_to_arg_string(val_names),
        "test_score_used_for_selection": test_score,
        "val_score": val_score,
        "baseline_val_score": baseline_val_score,
    }
    for key, value in val_metrics.items():
        row[f"val_{key}"] = value
    for key, value in baseline_val_metrics.items():
        row[f"baseline_val_{key}"] = value
    return row


def prediction_rows(protocol, task, fold_id, candidate, split, names, actual, pred, baseline):
    rows = []
    for idx, name in enumerate(names):
        row = {
            "protocol": protocol,
            "task": task,
            "fold_id": fold_id,
            "split": split,
            "dataset": name,
            "feature_family": candidate["feature_family"],
            "n_bins": candidate["n_bins"],
            "model_kind": candidate["model_kind"],
            "regularization": candidate["regularization"],
        }
        if task == "jacobi_ab":
            row.update({
                "actual_a": actual[idx, 0],
                "actual_b": actual[idx, 1],
                "pred_a": pred[idx, 0],
                "pred_b": pred[idx, 1],
                "baseline_pred_a": baseline[idx, 0],
                "baseline_pred_b": baseline[idx, 1],
                "abs_error_a": abs(actual[idx, 0] - pred[idx, 0]),
                "abs_error_b": abs(actual[idx, 1] - pred[idx, 1]),
                "baseline_abs_error_a": abs(actual[idx, 0] - baseline[idx, 0]),
                "baseline_abs_error_b": abs(actual[idx, 1] - baseline[idx, 1]),
            })
        else:
            row.update({
                "actual": actual[idx],
                "pred": pred[idx],
                "baseline_pred": baseline[idx],
                "correct": actual[idx] == pred[idx],
                "baseline_correct": actual[idx] == baseline[idx],
            })
        rows.append(row)
    return rows


def task_dataset_names(task, target_table, metric_ok_names):
    if task == "jacobi_ab":
        mask = target_table["has_jacobi_target"]
    else:
        mask = target_table["has_paper_target"]
    names = [
        row["dataset"]
        for _, row in target_table[mask].iterrows()
        if row["dataset"] in metric_ok_names
    ]
    return sorted(names)


def evaluate_nested_fold(task, fold_id, train_names, test_names, val_names,
                         target_table, feature_vectors, candidates, write_candidate_scores):
    fold_row = {
        "protocol": "nested_2_test_2_val",
        "task": task,
        "fold_id": fold_id,
        "train_datasets": values_to_arg_string(train_names),
        "test_datasets": values_to_arg_string(test_names),
        "val_datasets": values_to_arg_string(val_names),
        "n_train": len(train_names),
        "n_test": len(test_names),
        "n_val": len(val_names),
    }

    candidate_results = []
    for candidate_id, candidate in enumerate(candidates):
        train_score = np.nan
        if write_candidate_scores:
            train_score, _, _, _, _, _, _ = evaluate_candidate(
                task, candidate, train_names, train_names, target_table, feature_vectors
            )
        test_score, base_test_score, _, _, _, _, _ = evaluate_candidate(
            task, candidate, train_names, test_names, target_table, feature_vectors
        )
        candidate_results.append((candidate_id, candidate, train_score, test_score, base_test_score))

    best = max(candidate_results, key=lambda item: (item[3], -item[0]))
    candidate_score_rows = []
    if write_candidate_scores:
        for candidate_id, candidate, train_score, test_score, base_test_score in candidate_results:
            candidate_score_rows.append(candidate_row(
                "nested_2_test_2_val",
                task,
                fold_id,
                candidate_id,
                candidate,
                train_names,
                test_names,
                val_names,
                train_score,
                test_score,
                base_test_score,
                selected=(candidate_id == best[0]),
            ))

    candidate_id, candidate, _, test_score, _ = best
    val_score, base_val_score, val_metrics, base_val_metrics, pred, base, actual = evaluate_candidate(
        task, candidate, train_names, val_names, target_table, feature_vectors
    )
    selected = selected_row(
        "nested_2_test_2_val",
        task,
        fold_id,
        candidate_id,
        candidate,
        train_names,
        test_names,
        val_names,
        test_score,
        val_score,
        base_val_score,
        val_metrics,
        base_val_metrics,
    )
    preds = prediction_rows(
        "nested_2_test_2_val", task, fold_id, candidate, "val", val_names, actual, pred, base
    )
    return fold_row, selected, preds, candidate_score_rows


def evaluate_train_val_fold(task, fold_id, train_names, val_names,
                            target_table, feature_vectors, candidates, write_candidate_scores):
    fold_row = {
        "protocol": "train_fit_2_val",
        "task": task,
        "fold_id": fold_id,
        "train_datasets": values_to_arg_string(train_names),
        "test_datasets": "",
        "val_datasets": values_to_arg_string(val_names),
        "n_train": len(train_names),
        "n_test": 0,
        "n_val": len(val_names),
    }

    candidate_results = []
    for candidate_id, candidate in enumerate(candidates):
        train_score, base_train_score, _, _, _, _, _ = evaluate_candidate(
            task, candidate, train_names, train_names, target_table, feature_vectors
        )
        candidate_results.append((candidate_id, candidate, train_score, base_train_score))

    best = max(candidate_results, key=lambda item: (item[2], -item[0]))
    candidate_score_rows = []
    if write_candidate_scores:
        for candidate_id, candidate, train_score, base_train_score in candidate_results:
            candidate_score_rows.append(candidate_row(
                "train_fit_2_val",
                task,
                fold_id,
                candidate_id,
                candidate,
                train_names,
                [],
                val_names,
                train_score,
                train_score,
                base_train_score,
                selected=(candidate_id == best[0]),
            ))

    candidate_id, candidate, train_score, _ = best
    val_score, base_val_score, val_metrics, base_val_metrics, pred, base, actual = evaluate_candidate(
        task, candidate, train_names, val_names, target_table, feature_vectors
    )
    selected = selected_row(
        "train_fit_2_val",
        task,
        fold_id,
        candidate_id,
        candidate,
        train_names,
        [],
        val_names,
        train_score,
        val_score,
        base_val_score,
        val_metrics,
        base_val_metrics,
    )
    preds = prediction_rows(
        "train_fit_2_val", task, fold_id, candidate, "val", val_names, actual, pred, base
    )
    return fold_row, selected, preds, candidate_score_rows


def run_jobs(jobs, n_jobs, backend):
    if n_jobs == 1 or len(jobs) <= 1:
        return [job() for job in jobs]
    try:
        return Parallel(n_jobs=n_jobs, backend=backend)(delayed(job)() for job in jobs)
    except PermissionError as exc:
        if backend != "loky":
            raise
        print(
            f"[cv] loky backend unavailable ({exc}); falling back to threading",
            flush=True,
        )
        return Parallel(n_jobs=n_jobs, backend="threading")(delayed(job)() for job in jobs)


def run_nested_cv(task, names, target_table, feature_vectors, candidates, streams,
                  max_val_pairs=None, max_test_pairs=None, n_jobs=1, parallel_backend="loky"):
    selected_rows = []
    pred_rows = []
    fold_rows = []
    val_pairs = list(combinations(names, 2))
    if max_val_pairs:
        val_pairs = val_pairs[:max_val_pairs]

    fold_id = 0
    for val_pair_index, val_pair in enumerate(val_pairs, start=1):
        remaining_after_val = [name for name in names if name not in val_pair]
        test_pairs = list(combinations(remaining_after_val, 2))
        if max_test_pairs:
            test_pairs = test_pairs[:max_test_pairs]
        print(
            f"[cv] {task} nested val pair {val_pair_index}/{len(val_pairs)}: "
            f"val={values_to_arg_string(val_pair)}, test_pairs={len(test_pairs)}, "
            f"candidates={len(candidates)}",
            flush=True,
        )
        write_candidate_scores = streams.get("candidate_scores") is not None
        jobs = []
        for test_pair in test_pairs:
            train_names = [name for name in remaining_after_val if name not in test_pair]
            test_names = list(test_pair)
            val_names = list(val_pair)
            fold_id += 1
            jobs.append(lambda task=task, fold_id=fold_id, train_names=train_names,
                        test_names=test_names, val_names=val_names,
                        write_candidate_scores=write_candidate_scores: evaluate_nested_fold(
                task,
                fold_id,
                train_names,
                test_names,
                val_names,
                target_table,
                feature_vectors,
                candidates,
                write_candidate_scores,
            ))

        for fold_row, selected, preds, candidate_score_rows in run_jobs(jobs, n_jobs, parallel_backend):
            fold_rows.append(fold_row)
            selected_rows.append(selected)
            pred_rows.extend(preds)
            if streams.get("candidate_scores") is not None:
                for row in candidate_score_rows:
                    streams["candidate_scores"].write(row)
    return fold_rows, selected_rows, pred_rows


def run_train_val_diagnostic(task, names, target_table, feature_vectors, candidates, streams,
                             max_val_pairs=None, n_jobs=1, parallel_backend="loky"):
    selected_rows = []
    pred_rows = []
    fold_rows = []
    val_pairs = list(combinations(names, 2))
    if max_val_pairs:
        val_pairs = val_pairs[:max_val_pairs]

    jobs = []
    write_candidate_scores = streams.get("candidate_scores") is not None
    for fold_id, val_pair in enumerate(val_pairs, start=1):
        print(
            f"[cv] {task} train-fit val pair {fold_id}/{len(val_pairs)}: "
            f"val={values_to_arg_string(val_pair)}, candidates={len(candidates)}",
            flush=True,
        )
        val_names = list(val_pair)
        train_names = [name for name in names if name not in val_pair]
        jobs.append(lambda task=task, fold_id=fold_id, train_names=train_names,
                    val_names=val_names,
                    write_candidate_scores=write_candidate_scores: evaluate_train_val_fold(
            task,
            fold_id,
            train_names,
            val_names,
            target_table,
            feature_vectors,
            candidates,
            write_candidate_scores,
        ))

    for fold_row, selected, preds, candidate_score_rows in run_jobs(jobs, n_jobs, parallel_backend):
        fold_rows.append(fold_row)
        selected_rows.append(selected)
        pred_rows.extend(preds)
        if streams.get("candidate_scores") is not None:
            for row in candidate_score_rows:
                streams["candidate_scores"].write(row)
    return fold_rows, selected_rows, pred_rows


def summarize(selected_rows, prediction_rows):
    selected = pd.DataFrame(selected_rows)
    predictions = pd.DataFrame(prediction_rows)
    rows = []
    if selected.empty:
        return rows

    for (protocol, task), group in selected.groupby(["protocol", "task"]):
        row = {
            "protocol": protocol,
            "task": task,
            "folds": int(len(group)),
            "mean_val_score": float(group["val_score"].mean()),
            "std_val_score": float(group["val_score"].std(ddof=0)),
            "mean_baseline_val_score": float(group["baseline_val_score"].mean()),
            "selected_feature_family_mode": Counter(group["feature_family"]).most_common(1)[0][0],
            "selected_n_bins_mode": int(Counter(group["n_bins"]).most_common(1)[0][0]),
        }
        task_preds = predictions[
            (predictions["protocol"] == protocol)
            & (predictions["task"] == task)
        ]
        if task == "jacobi_ab" and not task_preds.empty:
            row.update({
                "mean_abs_error_a": float(task_preds["abs_error_a"].mean()),
                "mean_abs_error_b": float(task_preds["abs_error_b"].mean()),
                "baseline_mean_abs_error_a": float(task_preds["baseline_abs_error_a"].mean()),
                "baseline_mean_abs_error_b": float(task_preds["baseline_abs_error_b"].mean()),
                "r2_a_all_predictions": safe_r2(task_preds["actual_a"], task_preds["pred_a"]),
                "r2_b_all_predictions": safe_r2(task_preds["actual_b"], task_preds["pred_b"]),
            })
        elif not task_preds.empty:
            row.update({
                "accuracy_all_predictions": float(task_preds["correct"].mean()),
                "baseline_accuracy_all_predictions": float(task_preds["baseline_correct"].mean()),
            })
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", default="spectral_massive_slurm")
    parser.add_argument("--jacobi-root", default="jacobi_ab_sweep_massive")
    parser.add_argument("--out-dir", default="feature_aware_slp_results")
    parser.add_argument("--datasets", nargs="+", default=PRIMARY_DATASETS)
    parser.add_argument("--bins", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--feature-families", nargs="+", choices=FEATURE_FAMILIES, default=FEATURE_FAMILIES)
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--regression-alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--logistic-cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps, or auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--spectral-method",
        choices=["auto", "exact", "chebyshev", "lanczos"],
        default="auto",
        help="auto uses exact dense eigendecomposition below --max-dense-elements and Chebyshev band projectors above it.",
    )
    parser.add_argument(
        "--max-dense-elements",
        type=int,
        default=50_000_000,
        help="Dense n*n cutoff for exact mode in auto. 0 disables the cutoff.",
    )
    parser.add_argument("--chebyshev-order", type=int, default=64)
    parser.add_argument(
        "--no-chebyshev-jackson",
        dest="chebyshev_jackson",
        action="store_false",
        help="Disable Jackson damping for approximate Chebyshev band projectors.",
    )
    parser.set_defaults(chebyshev_jackson=True)
    parser.add_argument("--lanczos-steps", type=int, default=64)
    parser.add_argument("--lanczos-feature-probes", type=int, default=16)
    parser.add_argument("--lanczos-seed", type=int, default=0)
    parser.add_argument(
        "--label-scope",
        choices=["full", "train"],
        default="full",
        help="full uses all labels; train reconstructs seed train masks and averages label metrics over them.",
    )
    parser.add_argument(
        "--label-split-mode",
        choices=["paper_random", "jacobi_native_or_random"],
        default="paper_random",
        help="paper_random matches train_spectral_massive.py; jacobi_native_or_random matches jacobi_ab_sweep.py.",
    )
    parser.add_argument(
        "--label-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds used for train-label SLP. Defaults to 0..expected-runs-1.",
    )
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--max-test-pairs", type=int, default=None)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["nested", "train_fit"],
        default=["nested", "train_fit"],
    )
    parser.add_argument(
        "--no-candidate-scores",
        dest="write_candidate_scores",
        action="store_false",
        help="Skip the very large candidate_scores.csv table for faster exploratory runs.",
    )
    parser.set_defaults(write_candidate_scores=True)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel CV fitting jobs. Use -1 for all cores.")
    parser.add_argument(
        "--parallel-backend",
        choices=["loky", "threading"],
        default="loky",
        help="joblib backend for CV fitting. loky uses processes; threading avoids process startup overhead.",
    )
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=["jacobi_ab", "architecture", "paper_K"])
    return parser.parse_args()


def main():
    args = parse_args()
    print("[startup] parsed args", flush=True)
    load_runtime_dependencies()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    paper_root = resolve_repo_path(args.paper_root)
    jacobi_root = resolve_repo_path(args.jacobi_root)
    out_dir = ensure_dir(resolve_repo_path(args.out_dir) / timestamp)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = resolve_device(args.device)
    if device.type == "mps" and dtype == torch.float64:
        print("[device] MPS does not support float64 well; using float32", flush=True)
        dtype = torch.float32
    if args.spectral_method == "lanczos":
        unsupported = set(args.feature_families) & {"band_cka", "combined"}
        if unsupported:
            raise ValueError(
                "Lanczos mode estimates spectral masses, not band projections. "
                f"Remove unsupported feature families: {sorted(unsupported)}"
            )
    datasets = [canonical_dataset_name(name) for name in args.datasets]
    label_seeds = args.label_seeds
    if label_seeds is None:
        label_seeds = list(range(args.expected_runs))

    config = vars(args).copy()
    config["resolved_label_seeds"] = label_seeds
    config["resolved_paper_root"] = str(paper_root)
    config["resolved_jacobi_root"] = str(jacobi_root)
    config["resolved_out_dir"] = str(out_dir)
    config["resolved_device"] = device.type
    config["resolved_dtype"] = str(dtype)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    paper_targets, _ = load_paper_targets(paper_root, args.expected_runs)
    jacobi_targets, _ = load_jacobi_targets(jacobi_root)
    target_table = build_target_table(datasets, paper_targets, jacobi_targets)
    target_table.to_csv(out_dir / "target_table.csv", index=False)

    dataset_map = load_dataset_map(datasets)
    dataset_map = {canonical_dataset_name(name): dataset for name, dataset in dataset_map.items()}
    metric_rows = []
    feature_vectors = {}
    metric_status = {}
    metadata_rows = []

    print(f"Output directory: {out_dir}", flush=True)
    for dataset_name in datasets:
        if dataset_name not in dataset_map:
            metric_status[dataset_name] = {"ok": False, "reason": "dataset_not_found_in_loader"}
            print(f"[metrics] {dataset_name}: missing from loader", flush=True)
            continue
        try:
            num_nodes = int(dataset_map[dataset_name][0].num_nodes)
            method_used = select_spectral_method(
                args.spectral_method,
                num_nodes,
                args.max_dense_elements,
            )
            if method_used == "exact":
                method_message = "exact eigensystem"
            elif method_used == "chebyshev":
                method_message = (
                    f"Chebyshev spectral projectors "
                    f"(order={args.chebyshev_order}, jackson={args.chebyshev_jackson})"
                )
            else:
                method_message = (
                    f"Lanczos spectral measures "
                    f"(steps={args.lanczos_steps}, feature_probes={args.lanczos_feature_probes})"
                )
            compute_device, fallback_reason = metric_compute_device(device, method_used)
            if fallback_reason:
                method_message += f" on {compute_device.type} ({fallback_reason})"
            else:
                method_message += f" on {compute_device.type}"
            print(f"[metrics] {dataset_name}: computing {method_message}", flush=True)
            rows, vectors, metadata = compute_dataset_metrics(
                dataset_name,
                dataset_map[dataset_name],
                args.bins,
                dtype=dtype,
                device=device,
                max_dense_elements=args.max_dense_elements,
                spectral_method=args.spectral_method,
                chebyshev_order=args.chebyshev_order,
                chebyshev_jackson=args.chebyshev_jackson,
                label_scope=args.label_scope,
                label_seeds=label_seeds,
                label_split_mode=args.label_split_mode,
                lanczos_steps=args.lanczos_steps,
                lanczos_feature_probes=args.lanczos_feature_probes,
                lanczos_seed=args.lanczos_seed,
                feature_families=args.feature_families,
            )
            metric_rows.extend(rows)
            feature_vectors.update(vectors)
            metadata_rows.append({"dataset": dataset_name, **metadata})
            metric_status[dataset_name] = {"ok": True, "reason": "computed"}
        except Exception as exc:
            metric_status[dataset_name] = {"ok": False, "reason": repr(exc)}
            print(f"[metrics] {dataset_name}: failed: {exc}", flush=True)

    write_csv(out_dir / "computed_metric_profiles.csv", metric_rows)
    write_csv(out_dir / "dataset_coverage.csv", coverage_rows(datasets, metric_status, target_table))
    write_csv(out_dir / "dataset_metadata.csv", metadata_rows)

    if args.metrics_only:
        print("metrics-only run complete", flush=True)
        return

    candidate_fields = [
        "protocol", "selection_split", "task", "fold_id", "candidate_id", "feature_family",
        "n_bins", "model_kind", "regularization", "train_datasets",
        "test_datasets", "val_datasets", "train_score", "test_score",
        "baseline_test_score", "selected",
    ]
    streams = {
        "candidate_scores": (
            CsvStream(out_dir / "candidate_scores.csv", candidate_fields)
            if args.write_candidate_scores
            else None
        )
    }

    all_fold_rows = []
    all_selected_rows = []
    all_prediction_rows = []
    metric_ok_names = {name for name, status in metric_status.items() if status["ok"]}

    for task in args.tasks:
        names = task_dataset_names(task, target_table, metric_ok_names)
        if len(names) < 5:
            print(f"[cv] {task}: skipping, only {len(names)} usable datasets", flush=True)
            continue
        candidates = make_candidates(
            task,
            args.bins,
            args.regression_alphas,
            args.logistic_cs,
            args.feature_families,
        )
        print(
            f"[cv] {task}: {len(names)} datasets, {len(candidates)} candidates",
            flush=True,
        )
        if "nested" in args.protocols:
            fold_rows, selected_rows, pred_rows = run_nested_cv(
                task,
                names,
                target_table,
                feature_vectors,
                candidates,
                streams,
                max_val_pairs=args.max_val_pairs,
                max_test_pairs=args.max_test_pairs,
                n_jobs=args.n_jobs,
                parallel_backend=args.parallel_backend,
            )
            all_fold_rows.extend(fold_rows)
            all_selected_rows.extend(selected_rows)
            all_prediction_rows.extend(pred_rows)

        if "train_fit" in args.protocols:
            fold_rows, selected_rows, pred_rows = run_train_val_diagnostic(
                task,
                names,
                target_table,
                feature_vectors,
                candidates,
                streams,
                max_val_pairs=args.max_val_pairs,
                n_jobs=args.n_jobs,
                parallel_backend=args.parallel_backend,
            )
            all_fold_rows.extend(fold_rows)
            all_selected_rows.extend(selected_rows)
            all_prediction_rows.extend(pred_rows)

    if streams["candidate_scores"] is not None:
        streams["candidate_scores"].close()
    else:
        write_csv(out_dir / "candidate_scores.csv", [], candidate_fields)

    write_csv(out_dir / "fold_assignments.csv", all_fold_rows)
    write_csv(out_dir / "selected_fold_results.csv", all_selected_rows)
    prediction_df = pd.DataFrame(all_prediction_rows)
    if not prediction_df.empty:
        prediction_df[prediction_df["task"] == "jacobi_ab"].to_csv(
            out_dir / "ab_predictions.csv",
            index=False,
        )
        prediction_df[prediction_df["task"] == "architecture"].to_csv(
            out_dir / "architecture_predictions.csv",
            index=False,
        )
        prediction_df[prediction_df["task"] == "paper_K"].to_csv(
            out_dir / "k_predictions.csv",
            index=False,
        )
    else:
        prediction_base_fields = [
            "protocol", "task", "fold_id", "split", "dataset", "feature_family",
            "n_bins", "model_kind", "regularization",
        ]
        write_csv(
            out_dir / "ab_predictions.csv",
            [],
            prediction_base_fields + [
                "actual_a", "actual_b", "pred_a", "pred_b",
                "baseline_pred_a", "baseline_pred_b", "abs_error_a",
                "abs_error_b", "baseline_abs_error_a", "baseline_abs_error_b",
            ],
        )
        write_csv(
            out_dir / "architecture_predictions.csv",
            [],
            prediction_base_fields + [
                "actual", "pred", "baseline_pred", "correct", "baseline_correct",
            ],
        )
        write_csv(
            out_dir / "k_predictions.csv",
            [],
            prediction_base_fields + [
                "actual", "pred", "baseline_pred", "correct", "baseline_correct",
            ],
        )

    summary_rows = summarize(all_selected_rows, all_prediction_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    if summary_rows:
        print("\nSummary", flush=True)
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"\nWrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
