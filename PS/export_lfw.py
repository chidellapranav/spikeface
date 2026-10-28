"""Host-only export: python -m PS.export_lfw --repo ../trmp_spike --output board_bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from app.lfw import load_reference
    from stress_sim.lfw_reference import Network, export_params
    _, state, labels, metadata = load_reference(args.repo, args.checkpoint)
    if not metadata['hardware_compatible']:
        raise ValueError('Checkpoint is incompatible with the LFW hardware')
    params = export_params(state)
    data = Path(args.repo) / 'data/spikeface_lfw'
    images = np.load(data / 'X_test.npy', allow_pickle=False).astype(np.float32)
    targets = np.load(data / 'y_test.npy', allow_pickle=False)
    if images.shape != (len(targets), 64, 64) or len(images) < 2:
        raise ValueError('Need at least two 64x64 test images')
    if not np.isfinite(images).all() or np.any((images < 0) | (images > 1)):
        raise ValueError('Invalid image pixels')
    if targets.dtype.kind not in 'iu' or targets.ndim != 1 or np.any((targets < 0) | (targets >= 53)):
        raise ValueError('Invalid dataset labels')
    if np.load(data / 'class_names.npy', allow_pickle=False).tolist() != labels:
        raise ValueError('Dataset label ordering differs from checkpoint')
    second = next((i for i in range(1, len(images)) if not np.array_equal(images[0], images[i])), None)
    if second is None:
        raise ValueError('Need two distinct images for reset verification')
    indices = [0, second]
    traces = []
    for index in indices:
        reference = Network(params, metadata['hls_config'])
        traces.append(np.stack([reference.step(images[index], clear=t == 0)['scores']
                                for t in range(metadata['timesteps'])]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = output / 'lfw_bundle.npz'
    if payload.exists() or (output / 'manifest.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    np.savez(payload, **params, images=images, targets=targets,
             smoke_indices=np.asarray(indices), expected_traces=np.asarray(traces, dtype=np.float32))
    metadata.update(format_version=1, reference='sequential float32 scalar contract',
                    payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Exported', output.resolve())


if __name__ == '__main__':
    main()
