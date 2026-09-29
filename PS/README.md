# PS — PYNQ driver for SpikeFace

`spikeface_driver.py` is the Processing-System side of SpikeFace. It runs on the
PYNQ-Z2's ARM core (Python + the `pynq` library) and drives the six-block PL
datapath. It contains both the driver and the timestep scheduler; there is no
separate scheduler module.

> **Status: written against the committed design, not yet run on hardware.**
> Its logic is unit-tested against a mock of the board (weight encoding, packing,
> score decoding, address-window check, start order, timeout path). That does not
> prove real-hardware behaviour. Expect a short bring-up round.
> Read `DEPLOYMENT_AND_ACCURACY.md` first: it lists two problems in the committed
> hardware that must be fixed before any accuracy number is meaningful.

## What the script does

| Piece | What it does |
|---|---|
| `SpikeFaceOverlay.__init__` | Loads the bitstream, binds the six IPs and two DMAs by their Vivado instance names, allocates the DMA buffers, and prints the fabric clock the design actually runs at. |
| `configure(weights, beta, threshold)` | One-time setup. Converts quantized weights to raw 8-bit two's-complement codes, writes them into each block's AXI-Lite weight array, writes `beta`/`threshold` to conv1, conv2, fc1, and reads every weight array back to verify it landed. |
| `encode_frame` | Bernoulli rate coding on the PS: a pixel of intensity p spikes with probability p. Same scheme as `spikegen.rate()` in the Python reference. |
| `_run_one_timestep` (the scheduler) | Waits for the four compute blocks to be idle, sets `clear_accum`, starts the blocks **downstream-first** (fc2, fc1, conv2, conv1), arms the output DMA, sends the spike frame through the input DMA, and waits for both to finish. Every wait has a timeout that reports which block or DMA is stuck. |
| `run_inference(image01 / spike_frames, T)` | Runs T timesteps, sets `clear_accum` only on the first, decodes fc2's 32-bit scores, returns `(argmax, scores)`. Optional: per-timestep score trace, per-layer sparsity, `fresh_state=True`. |
| `reset_pl_state()` | Reprograms the PL and rewrites all parameters so no membrane state carries over from the previous image (see the deployment doc). |
| `read_active_taps()` | Reads each spiking block's `total_active_taps` counter, used for measured sparsity. |

## Why downstream-first

Each compute block only drains its input stream once it has been started. Starting
upstream-first lets conv1 push data through the free-running pooling block into
conv2's FIFO before conv2 is listening; the FIFO fills and everything stalls. The
two pooling blocks need no start: their `ap_start` is tied high in hardware.

## Data formats the driver expects

| Item | Format |
|---|---|
| Weights | Already-quantized values (the `w_*_q` arrays from the Python fixed-point reference), not raw float weights. Off-grid values are rejected. |
| Weight shapes | conv1 `(8,5,5)` (not `(8,1,5,5)`), conv2 `(16,8,5,5)`, fc1 `(64,1024)`, fc2 `(40,64)` |
| Weight formats | conv1/conv2/fc1: `ap_fixed<8,1>` (7 fractional bits). fc2: `ap_fixed<8,2>` (6 fractional bits). |
| `beta`, `threshold` | Real values (`0.9375`, `1.0`); encoded as `ap_fixed<8,3>` (codes 30 and 32). |
| Image | 32x32 floats in [0,1] |
| Replay input | `spike_frames`: `(T,32,32)` or `(T,1024)` of 0/1, to replay exact spikes |
| Scores | fc2 returns `ap_fixed<32,8>` bit patterns; decoded to floats here |

## Register and address facts used (read from the committed XSA / `.hwh`)

| Block | AXI-Lite base | Weights offset | Notes |
|---|---|---|---|
| conv1_layer_axi_0 | `0x40000000` | `0x100` (200 B) | |
| conv2_layer_axi_0 | `0x40010000` | `0x1000` (3200 B) | |
| fc1_layer_axi_0 | `0x40020000` | `0x10000` (64 KiB) | **window too small in the committed build; see the deployment doc** |
| fc2_layer_axi_0 | `0x40030000` | `0x1000` (2560 B) | `clear_accum` at `0x10` |
| axi_dma_0 / axi_dma_1 | `0x41E00000` / `0x41E10000` | | MM2S 8-bit / S2MM 32-bit, Direct Register mode |

If the block design changes, these come from the regenerated `.hwh` and the
`x<block>_hw.h` headers inside the XSA; update `WEIGHT_OFFSET` to match.

## Minimal use

```python
from spikeface_driver import SpikeFaceOverlay
import numpy as np

sf = SpikeFaceOverlay("/home/xilinx/spikeface/spikeface.bit")
sf.configure({k: np.load(f"w_{k}_q.npy") for k in ("conv1", "conv2", "fc1", "fc2")})
pred, scores = sf.run_inference(image01, T=50)
```

## Not in this folder yet

- Scripts to export the quantized weights as `.npy`, the spike frames and expected
  scores for the golden replay, and the 80-image test set.
- An evaluation loop. The method is in `DEPLOYMENT_AND_ACCURACY.md`.
