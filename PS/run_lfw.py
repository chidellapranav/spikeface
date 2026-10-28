"""Board-only runner; requires NumPy and PYNQ, without PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def load_bundle(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'manifest.json').read_text())
    payload = folder / 'lfw_bundle.npz'
    if metadata['format_version'] != 1 or hashlib.sha256(payload.read_bytes()).hexdigest() != metadata['payload_sha256']:
        raise ValueError('Bundle version or SHA-256 mismatch')
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def smoke(board, arrays, timesteps):
    records = []
    # A -> B -> A verifies different-image reset and deterministic replay.
    for slot in (0, 1, 0):
        index = int(arrays['smoke_indices'][slot])
        pred, scores, trace = board.infer(arrays['images'][index], timesteps, True)
        expected = arrays['expected_traces'][slot]
        passed = trace.shape == expected.shape and np.array_equal(trace.view(np.uint32), expected.view(np.uint32))
        records.append({'index': index, 'prediction': pred, 'passed': bool(passed),
                        'max_absolute_error': float(np.max(np.abs(trace - expected)))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--bitstream', required=True)
    parser.add_argument('--mode', choices=['smoke', 'evaluate'], default='smoke')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--output', default='board_results.json')
    args = parser.parse_args()
    bit = Path(args.bitstream)
    if bit.suffix != '.bit' or not bit.is_file() or not bit.with_suffix('.hwh').is_file():
        parser.error('Provide matching .bit/.hwh files with the same basename')
    metadata, arrays = load_bundle(args.bundle)
    if args.limit is not None and not 1 <= args.limit <= len(arrays['images']):
        parser.error('--limit must be within the dataset size')
    from PS.lfw_driver import LFWOverlay
    board = LFWOverlay(str(bit), {k: arrays[k] for k in ('conv1', 'conv2', 'fc1')},
                       timeout_s=args.timeout, config=metadata['hls_config'])
    report = {'bundle_sha256': metadata['payload_sha256'],
              'bitstream_sha256': hashlib.sha256(bit.read_bytes()).hexdigest(),
              'hwh_sha256': hashlib.sha256(bit.with_suffix('.hwh').read_bytes()).hexdigest(),
              'checkpoint_sha256': metadata['checkpoint_sha256'], 'fclk_mhz': board.fclk_mhz}
    try:
        report['smoke'] = smoke(board, arrays, metadata['timesteps'])
        report['passed'] = all(record['passed'] for record in report['smoke'])
        if not report['passed']:
            raise RuntimeError('Golden trace mismatch; dataset evaluation stopped')
        if args.mode == 'evaluate':
            count = args.limit or len(arrays['images'])
            predictions, scores = [], []
            start = time.perf_counter()
            for index, image in enumerate(arrays['images'][:count]):
                pred, score = board.infer(image, metadata['timesteps'])
                if not np.isfinite(score).all():
                    raise RuntimeError('Nonfinite hardware scores')
                predictions.append(pred)
                scores.append(score.tolist())
                print('Image {}/{}: {}'.format(index + 1, count, metadata['classes'][pred]), flush=True)
            targets = arrays['targets'][:count]
            confusion = np.zeros((53, 53), dtype=np.int64)
            np.add.at(confusion, (targets, predictions), 1)
            report.update(images=count, correct=int(np.trace(confusion)),
                          accuracy=float(np.trace(confusion) / count),
                          partial_evaluation=count != len(arrays['images']),
                          seconds=time.perf_counter() - start, predictions=predictions,
                          targets=targets.tolist(), scores=scores,
                          classes=metadata['classes'], confusion_matrix=confusion.tolist())
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        board.close()
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
