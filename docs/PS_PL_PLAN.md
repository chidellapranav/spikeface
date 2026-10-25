# Authoritative LFW PS and PL integration plan

The requested authority is `trmp_spike`, commit
`7a4d1f4`, and the physical timing requirement is **WNS ≥ 0 ns**.
The old 32×32 / 40-class XSA cannot execute this model.

## Contract to freeze first

| Stage | Input | Computation | Output |
|---|---|---|---|
| `lfw_conv1_axi` | 4096 float32 grayscale values | 1→8, 5×5 convolution with bias; BN affine; 2×2 max pool; delayed subtractive LIF | 8192 byte-wide binary spikes |
| `lfw_conv2_axi` | 8192 byte-wide spikes | 8→16, 5×5 convolution with bias; BN affine; max pool; delayed subtractive LIF | 4096 byte-wide binary spikes |
| `lfw_readout_axi` | 4096 byte-wide spikes | 4096→53 linear with bias; BN affine; leaky membrane integration; running mean of membrane trace | 53 IEEE float32 scores with TLAST on score 53 |

Input is the SAME normalized static image at every timestep, not rate-coded
binary input. Default T=25, beta=.9375, hidden threshold=.5. Hidden reset is
`mem_new = beta*mem_old + current - (mem_old > threshold ? threshold : 0)`;
spikes use strict `mem_new > threshold`. The readout has no reset-on-spike.
Dropout is disabled in inference. BN is an inference affine transformation,
applied after convolution/linear output and before max pooling. Conv kernels
are not folded with BN: this avoids changing their accumulation order further.

Float32 HLS and PyTorch may use different reduction orders; bitwise equivalence
to arbitrary PyTorch kernels is not assumed. The independent scalar contract,
full dataset predictions, and numerical divergence must all be reported before
hardware acceptance. See the LFW verification report for the current verdict.

## Processing system

`app.lfw` already supplies dataset inference for the authoritative model.
`PS/lfw_driver.py` provides the matching board-side scheduler candidate. It has
not been run on the board and must not be confused with a validated runtime.

1. Load matching `.bit` and `.hwh`; verify three LFW instance names and DMA widths.
2. Export weights, biases and BN affine arrays from the locked checkpoint. Pack
   each stage as weights, bias, BN scale, BN shift. Allocate 224, 3248 and 217247
   float32 elements in contiguous DDR. Flush and program each 64-bit pointer.
3. Allocate input `(4096,) float32` and output `(53,) float32`. The readout weights
   alone exceed the chip's on-chip memory budget, so parameter DDR masters are
   intentional; do not reproduce the legacy AXI-Lite weight-window design.
4. On timestep zero set every clear_state=1; subsequent timesteps set all to zero.
   Start readout→conv2→conv1, arm S2MM, then MM2S. Repeat the same image T times.
5. Poll with timeouts and DMA error checks; call PYNQ completion housekeeping so
   output caches are invalidated; require 16384 input and 212 output bytes.
6. Compare all score traces with exported expected traces before collecting
   dataset accuracy. Use fresh state for each image, identical image ordering,
   exact label mapping and locked preprocessing. Produce confusion matrix,
   per-class accuracy, model/data hashes and an accuracy confidence interval.

Offsets checked against native synthesis RTL: params low/high at 0x10/0x14,
beta at 0x1c; hidden threshold at 0x24 and clear at 0x2c; readout clear at 0x24.
Regenerated HWH/driver headers remain the final authority after interface edits.

PYNQ DMA completion source:
[PYNQ 3.0 DMA implementation](https://pynq.readthedocs.io/en/v3.0.0/_modules/pynq/lib/dma.html).
The legacy driver's idle-only path was repaired because it bypassed normal
receive completion housekeeping.

## Programmable logic, after verification acceptance

Use `build_lfw_hls.tcl` to build the three new tops. Keep the legacy rebuild
script and XSA clearly marked as historical. Start with the board's established
50 MHz fabric clock; raise it only after implementation passes timing. No
latency requirement is substituted for the user's nonnegative-WNS requirement.

The new block design needs:

- PYNQ-Z2 PS7 with its actual board DDR/IO preset, not a generic guessed preset.
- Two simple-mode DMAs: 32-bit MM2S input and 32-bit S2MM output.
- Three LFW HLS IPs, with stream chain DMA→conv1→conv2→readout→DMA.
- A GP0 AXI-Lite control interconnect with five masters/targets: the three HLS
  controls and two DMA control ports. Allocate nonoverlapping control ranges.
- DDR connectivity for FIVE memory masters: two DMA engines and three HLS PARAM
  masters. Use SmartConnect/arbitration into enabled PS high-performance ports.
- One coherent clock/reset domain initially; register slices/FIFOs only as
  justified by RTL scheduling and implementation reports.

Run C-simulation, synthesis, RTL co-simulation under stalls and back-to-back
images, IP export, BD validation, implementation, timing and resource checks,
then export matching bit/HWH. Require **routed WNS ≥ 0**, no unresolved critical
warnings or unconstrained functional paths, and resource fit. HLS estimated
periods and loop II reports do not establish routed timing. Reduced throughput
from floating-point accumulation dependencies is acceptable under the stated
latency preference, but does not waive timing closure.

## Accuracy development

Preserving the supplied checkpoint preserves its measured ~73.29% performance;
changing PS/PL timing cannot raise it to >96%. The original training script uses
the test set to select the best epoch, so that split is already a model-selection
set. Establish a fresh held-out set before a publication-grade final claim.

Train/validation work should retain the agreed architecture initially, select
checkpoints using validation only, match inference preprocessing and encoding,
and test quantization only after a floating-point baseline is accepted. The
original sparsity statistic uses `.item()` and has no gradient, so adding it to
the loss does not implement differentiable sparsity regularization. Correct
that in a controlled training experiment rather than attributing accuracy to
the FPGA. A 554-image set needs at least **532 correct** to exceed 96%; no such
result has been achieved by the supplied checkpoint.
