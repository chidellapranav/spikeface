import argparse
import json
from pathlib import Path
import numpy as np
from .engine import Model, evaluate, fixture_frames, image_frames


def main():
    p = argparse.ArgumentParser(description="SpikeFace labeled-dataset inference and HLS verification")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="replay the committed fixture; not an accuracy test")
    lfw = commands.add_parser("lfw-evaluate", help="measure the authoritative 53-class model on a labeled image dataset")
    lfw.add_argument("--repo", required=True)
    lfw.add_argument("--checkpoint")
    lfw.add_argument("--output", default="lfw_evaluation.json")
    lfw.add_argument("--batch-size", type=int, default=32)
    lfw.add_argument("--limit", type=int)
    lfw.add_argument("--backend", choices=["software", "hls", "pynq"], default="software")
    lfw.add_argument("--bitstream")
    datasets = lfw.add_mutually_exclusive_group()
    datasets.add_argument("--dataset-dir")
    datasets.add_argument("--images-root")
    export = commands.add_parser("export-fixture")
    export.add_argument("output")
    infer = commands.add_parser("infer")
    infer.add_argument("--weights", required=True)
    inputs = infer.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--spikes", help=".npy exact binary spike frames")
    infer.add_argument("--timesteps", type=int, default=50)
    infer.add_argument("--seed", type=int, default=0)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--weights", required=True)
    ev.add_argument("--dataset", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--weights")
    serve.add_argument("--port", type=int, default=8080)
    for command in (infer, ev, serve):
        command.add_argument("--backend", choices=["software", "pynq"], default="software")
        command.add_argument("--bitstream", help="Matching .bit/.hwh base; required for PYNQ")
    args = p.parse_args()
    if args.command == "lfw-evaluate":
        from .lfw import evaluate_lfw
        evaluate_lfw(args.repo, args.checkpoint, args.batch_size, args.output, args.limit, args.dataset_dir, args.images_root, args.backend, args.bitstream)
        return
    if args.command == "export-fixture":
        Model.golden().save(args.output)
        print(f"Saved verification fixture: {Path(args.output).resolve()}")
        return
    model = Model.load(args.weights) if getattr(args, "weights", None) else Model.golden()
    if getattr(args, "backend", "software") == "pynq":
        if not args.bitstream:
            p.error("--bitstream is required for --backend pynq")
        from .pynq_backend import PynqModel
        model = PynqModel(model, args.bitstream)
    if args.command == "serve":
        from .server import serve
        serve(model, args.port)
        return
    try:
        if args.command == "evaluate":
            result = evaluate(model, args.dataset)
        else:
            frames = fixture_frames() if args.command == "demo" else (
                np.load(args.spikes, allow_pickle=False) if args.spikes else image_frames(args.image, args.timesteps, args.seed))
            result = model.infer(frames)
            result.pop("trace_codes")
        print(json.dumps(result, indent=2))
    finally:
        if hasattr(model, "close"):
            model.close()


if __name__ == "__main__":
    main()
