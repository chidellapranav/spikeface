"""Validation-selected readout refit; preserves LFW architecture and hidden layers.

The supplied checkpoint was already selected using its test split. This experiment
does not erase that provenance and cannot create a new independent holdout.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .lfw import load_reference


def features(model, images, timesteps=25, batch_size=32):
    import torch
    result = []
    beta = float(model.lif3.beta)
    coefficients = np.array([(1-beta**(timesteps-t))/(timesteps*(1-beta)) for t in range(timesteps)], dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i+batch_size]).unsqueeze(1)
            m1, m2 = model.lif1.init_leaky(), model.lif2.init_leaky()
            summed = torch.zeros((len(x), 4096))
            for coefficient in coefficients:
                s1, m1 = model.lif1(model.pool1(model.bn1(model.conv1(x))), m1)
                s2, m2 = model.lif2(model.pool2(model.bn2(model.conv2(s1))), m2)
                summed += float(coefficient)*s2.flatten(1)
            result.append(summed.numpy())
            print(f"Readout features: {min(i+batch_size,len(images))}/{len(images)}", flush=True)
    return np.concatenate(result), float(coefficients.sum())


def main():
    import torch
    from scipy.linalg import cho_factor, cho_solve
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    folder, repo = Path(args.output_dir), Path(args.repo)
    folder.mkdir(parents=True, exist_ok=True)
    model, state, labels, metadata = load_reference(repo)
    images = np.load(repo/"data/spikeface_lfw/X_train.npy", allow_pickle=False)
    targets = np.load(repo/"data/spikeface_lfw/y_train.npy", allow_pickle=False)
    start = time.perf_counter()
    x, coefficient_sum = features(model, images)
    rng, train, validation = np.random.default_rng(20261024), [], []
    for label in range(len(labels)):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        count = max(1, int(round(.2*len(indices))))
        validation.extend(indices[:count]);train.extend(indices[count:])
    train, validation = np.array(sorted(train)), np.array(sorted(validation))
    y = np.eye(53, dtype=np.float64)[targets]
    mean, scale = x[train].mean(0, dtype=np.float64), x[train].std(0, dtype=np.float64)
    scale = np.maximum(scale, .1)
    z = (x[train]-mean)/scale
    ymean = y[train].mean(0)
    kernel = z @ z.T
    identity = np.eye(len(train))
    rows, best = [], None
    for alpha in [0.01, .1, 1., 10., 100., 1000., 10000.]:
        dual = cho_solve(cho_factor(kernel+alpha*identity, lower=True), y[train]-ymean)
        coef = z.T @ dual
        bias = ymean - (mean/scale) @ coef
        weights = coef/scale[:, None]
        prediction = (x[validation] @ weights+bias).argmax(1)
        score = float(np.mean(prediction == targets[validation]))
        row = {"alpha": alpha, "validation_accuracy": score}
        rows.append(row)
        print(row, flush=True)
        if best is None or score > best[0]:
            best = score, alpha
    # Refit selected regularization on ALL training images; never choose on test.
    mean, scale = x.mean(0, dtype=np.float64), np.maximum(x.std(0, dtype=np.float64), .1)
    z = (x-mean)/scale
    ymean = y.mean(0)
    dual = cho_solve(cho_factor(z @ z.T+best[1]*np.eye(len(x)), lower=True), y-ymean)
    coef = z.T @ dual
    weights = coef/scale[:, None]
    bias = ymean-(mean/scale) @ coef
    new_state = {k: v.clone() for k, v in state.items()}
    new_state["fc1.weight"] = torch.tensor(weights.T, dtype=torch.float32)
    new_state["fc1.bias"] = torch.tensor(bias/coefficient_sum, dtype=torch.float32)
    new_state["bn3.running_mean"] = torch.zeros(53)
    new_state["bn3.running_var"] = torch.full((53,), 1-1e-5)
    new_state["bn3.weight"] = torch.ones(53)
    new_state["bn3.bias"] = torch.zeros(53)
    checkpoint = folder/"lfw_readout_refit.pt"
    torch.save(new_state, checkpoint)
    report = {"reference": metadata, "seed": 20261024, "method": "Frozen hidden layers; ridge readout selected on stratified training-only validation",
              "training_images": len(train), "validation_images": len(validation), "train_indices": train.tolist(),
              "validation_indices": validation.tolist(), "candidates": rows, "selected_alpha": best[1],
              "best_validation_accuracy": best[0], "checkpoint": str(checkpoint), "seconds": time.perf_counter()-start,
              "test_used_for_selection": False, "inherited_test_selection_caveat": "The frozen original checkpoint was previously selected on the supplied test split."}
    (folder/"readout_refit_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {checkpoint}; evaluate once with app lfw-evaluate --checkpoint", flush=True)


if __name__ == "__main__":
    main()
