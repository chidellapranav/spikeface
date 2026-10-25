"""Evaluate the user's authoritative LFW source without running its training script."""
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps


def load_reference(repo, checkpoint=None, threads=4):
    import torch
    import snntorch as snn
    from snntorch import surrogate
    repo = Path(repo).resolve()
    source = repo / "02_snn_model_train.py"
    tree = ast.parse(source.read_text())
    # Execute the original model class verbatim, excluding training, dataset
    # downloads, output-directory creation and global RNG changes.
    model_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpikingFaceNet")
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("BETA", "LIF_THRESHOLD", "NUM_STEPS")}
    namespace = {"torch": torch, "nn": torch.nn, "snn": snn, "surrogate": surrogate, **literals}
    exec(compile(ast.Module(body=[model_node], type_ignores=[]), str(source), "exec"), namespace)
    labels = np.load(repo/"data/spikeface_lfw/class_names.npy", allow_pickle=False)
    if labels.shape != (53,) or labels.dtype.kind not in "US":
        raise ValueError("Expected the authoritative 53-class LFW label map")
    model = namespace["SpikingFaceNet"](len(labels))
    checkpoint = Path(checkpoint) if checkpoint else repo/"phase1_outputs/spikeface_model.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite parameters")
    model.eval()
    torch.set_num_threads(threads)
    metadata = {"source_repo": str(repo), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "classes": labels.tolist(), "torch": torch.__version__, "snntorch": snn.__version__,
                "timesteps": literals["NUM_STEPS"], "encoding": "constant grayscale input; not Bernoulli"}
    metadata["hls_config"] = [float(model.lif1.beta.clamp(0, 1)), float(model.lif1.threshold),
                              float(model.lif2.beta.clamp(0, 1)), float(model.lif2.threshold),
                              float(model.lif3.beta.clamp(0, 1))]
    metadata["hardware_compatible"] = all(float(state[f"lif{i}.graded_spikes_factor"]) == 1
                                           and int(state[f"lif{i}.reset_mechanism_val"]) == reset
                                           for i, reset in [(1, 0), (2, 0), (3, 2)])
    return model, state, labels.tolist(), metadata


def evaluate_lfw(repo, checkpoint=None, batch_size=32, output="lfw_evaluation.json", limit=None,
                 dataset_dir=None, images_root=None, backend="software", bitstream=None):
    import torch
    model, state, labels, metadata = load_reference(repo, checkpoint)
    if backend not in ("software", "hls", "pynq"):
        raise ValueError("Unknown inference backend")
    if backend == "pynq" and not bitstream:
        raise ValueError("PYNQ requires a matching new LFW --bitstream")
    if backend != "software" and not metadata["hardware_compatible"]:
        raise ValueError("Checkpoint reset mechanisms or graded spikes are unsupported by the LFW HLS")
    repo = Path(repo)
    file_names = None
    if images_root:
        paths, arrays, targets = [], [], []
        root = Path(images_root)
        if not root.is_dir():
            raise ValueError("images_root must be a directory")
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name not in labels:
                raise ValueError(f"Unknown identity folder: {folder.name}; folder names must match class_names.npy")
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".pgm"):
                    continue
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("L")
                    if image.size != (64, 64):
                        raise ValueError(f"{path}: pre-align and resize to 64x64 using the training preprocessing first")
                    arrays.append(np.asarray(image, dtype=np.float32)/255)
                paths.append(path)
                targets.append(labels.index(folder.name))
        if not arrays:
            raise ValueError("No labeled 64x64 images found")
        x, y = np.stack(arrays), np.asarray(targets, dtype=np.int64)
        file_names = [str(p) for p in paths]
        dataset_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    else:
        folder = Path(dataset_dir) if dataset_dir else repo/"data/spikeface_lfw"
        names = np.load(folder/"class_names.npy", allow_pickle=False)
        if names.tolist() != labels:
            raise ValueError("Dataset class_names.npy must match the checkpoint's exact 53-class ordering")
        x_path, y_path = folder/"X_test.npy", folder/"y_test.npy"
        x, y = np.load(x_path, allow_pickle=False), np.load(y_path, allow_pickle=False)
        dataset_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [x_path, y_path]}
    if x.shape != (len(y), 64, 64) or y.ndim != 1 or y.dtype.kind not in "iu" or not np.all(np.isfinite(x)):
        raise ValueError("Invalid LFW dataset shapes or values")
    if np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y >= len(labels))):
        raise ValueError("LFW pixel/label range is invalid")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be in 1..256")
    if limit is not None:
        if not 1 <= limit <= len(y):
            raise ValueError("limit must be in 1..dataset size")
        x, y = x[:limit], y[:limit]
        if file_names is not None:
            file_names = file_names[:limit]
    predictions, logits = [], []
    start = time.perf_counter()
    board = None
    try:
        if backend == "software":
            with torch.inference_mode():
                for i in range(0, len(y), batch_size):
                    images = torch.from_numpy(x[i:i+batch_size]).unsqueeze(1)
                    seq = images.unsqueeze(0).repeat(metadata["timesteps"], 1, 1, 1, 1)
                    scores, _ = model(seq)
                    predictions.extend(scores.argmax(1).tolist())
                    logits.extend(scores.tolist())
                    print(f"LFW inference: {min(i+batch_size, len(y))}/{len(y)}", flush=True)
        else:
            from stress_sim.lfw_reference import export_params
            params = export_params(state)
            if backend == "hls":
                from stress_sim.lfw_verify import build, run_cpp
                build()
            else:
                from PS.lfw_driver import LFWOverlay
                board = LFWOverlay(bitstream, params, config=metadata["hls_config"])
            for i, image in enumerate(x):
                if backend == "hls":
                    frames = np.repeat(image[None], metadata["timesteps"], axis=0)
                    trace = run_cpp("chain", frames, params, [1]+[0]*(metadata["timesteps"]-1), metadata["hls_config"])
                    scores = trace[-1]["axi_scores"]
                    pred = int(np.argmax(scores))
                else:
                    pred, scores = board.infer(image, metadata["timesteps"])
                predictions.append(pred)
                logits.append(scores.astype(float).tolist())
                print(f"LFW {backend} inference: {i+1}/{len(y)}", flush=True)
    finally:
        if board is not None:
            board.close()
    predictions = np.asarray(predictions)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    correct, n = int(np.trace(confusion)), len(y)
    p, z = correct/n, 1.959963984540054
    center, half = (p+z*z/(2*n))/(1+z*z/n), z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    report = {"model": "authoritative LFW architecture", "backend": backend, "metadata": metadata,
              "dataset_sha256": dataset_hashes, "image_files": file_names,
              "images": n, "correct": correct, "accuracy": correct/n, "target_exceeded": correct/n > .96,
              "wilson_95": [center-half, center+half], "seconds": time.perf_counter()-start,
              "confusion_matrix": confusion.tolist(), "predictions": predictions.tolist(), "labels": y.tolist(),
              "logits": logits, "per_class": [{"label": label, "images": int(confusion[i].sum()),
                                               "correct": int(confusion[i, i]),
                                               "accuracy": float(confusion[i, i]/confusion[i].sum()) if confusion[i].sum() else None}
                                              for i, label in enumerate(labels)],
              "partial_evaluation": limit is not None,
              "selection_caveat": "For the bundled test split, original training selected checkpoints using this split; it is not an untouched final holdout."}
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ["images", "correct", "accuracy", "target_exceeded", "seconds"]}), flush=True)
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output", default="lfw_evaluation.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--backend", choices=["software", "hls", "pynq"], default="software")
    p.add_argument("--bitstream")
    datasets = p.add_mutually_exclusive_group()
    datasets.add_argument("--dataset-dir", help="Directory containing X_test.npy and y_test.npy")
    datasets.add_argument("--images-root", help="53-class label names as folders containing pre-aligned 64x64 images")
    args = p.parse_args()
    evaluate_lfw(args.repo, args.checkpoint, args.batch_size, args.output, args.limit, args.dataset_dir, args.images_root, args.backend, args.bitstream)


if __name__ == "__main__":
    main()
