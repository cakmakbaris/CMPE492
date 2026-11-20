"""Verify saved round-level vectors, annotation packets and patient sequences."""
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.utils import digest, read


def packet_path(source: Path, recorded_path: Path) -> Path:
    """Resolve packets consolidated beside the source manifest during archiving."""
    if (not recorded_path.is_file()
            and recorded_path.parent.parts[-4:] == ("results", "round_error_experiment", "run001", "packets")):
        return source / "packets" / recorded_path.name
    return recorded_path


def load_vectors(source: Path) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray]]:
    manifest = read(source / "manifest.json")
    if digest(source / "labels_reviewed.json") != manifest["annotation_sha256"]:
        raise ValueError("Canonical annotations changed")
    patients = manifest["patients"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    requests: set[str] = set()
    for index, patient in enumerate(patients, 1):
        vectors: list[np.ndarray] = []
        for row in patient["calls"]:
            folder = Path(row["capture"])
            if digest(folder / "activations.pt") != row["tensor_sha256"] or digest(folder / "metadata.json") != row["metadata_sha256"]:
                raise ValueError("Capture changed")
            if digest(packet_path(source, Path(row["packet"]))) != row["packet_sha256"]:
                raise ValueError("Annotation packet changed")
            metadata = read(folder / "metadata.json")
            if row["request_id"] in requests or metadata["request_id"] != row["request_id"] or metadata["capture_version"] != 3 or metadata["model"] != "qwen3.5-4b":
                raise ValueError("Capture identity mismatch")
            requests.add(row["request_id"])
            states = torch.load(folder / "activations.pt", map_location="cpu", weights_only=True)
            tensor = states["input"]["model.layers.21"]
            if tuple(tensor.shape) != (1, 2560) or not torch.isfinite(tensor).all():
                raise ValueError("Invalid endpoint vector: input/model.layers.21")
            vectors.append(tensor.float().numpy().reshape(-1))
        labels = np.array([-1 if r["label"] is None else r["label"] for r in patient["calls"]])
        if not (labels >= 0).any():
            raise ValueError("Unexpected all-unlabeled patient")
        xs.append(np.stack(vectors))
        ys.append(labels)
        if index % 40 == 0:
            print(f"Loaded {index}/{len(patients)} patients", flush=True)
    return patients, xs, ys


def preprocess(xs: list[np.ndarray], train: np.ndarray) -> tuple[StandardScaler, list[np.ndarray]]:
    scaler = StandardScaler().fit(np.concatenate([xs[i] for i in train]))
    return scaler, [scaler.transform(x).astype(np.float32) for x in xs]


def pack(xs: list[np.ndarray], ys: list[np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(xs[i]) for i in indices)
    x = torch.zeros(len(indices), width, xs[indices[0]].shape[-1])
    target = torch.zeros(len(indices), width)
    mask = torch.zeros(len(indices), width)
    for j, i in enumerate(indices):
        n = len(xs[i])
        x[j, :n] = torch.from_numpy(xs[i])
        target[j, :n] = torch.from_numpy(np.maximum(ys[i], 0).astype(np.float32))
        mask[j, :n] = torch.from_numpy((ys[i] >= 0).astype(np.float32))
    return x, target, mask
