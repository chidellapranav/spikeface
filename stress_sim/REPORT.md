# SpikeFace HLS functional verification report

**Scope update:** this report describes the legacy 32×32 / 40-class design.
After the owner supplied the Python repository, it proved to be an incompatible
64×64 / 53-class LFW model and was selected as the new authority. The current
verdict and measured accuracy are in [LFW_REPORT.md](LFW_REPORT.md). Statements
below about a missing original reference refer to the unrecovered ORL/golden
export workflow, not the now-available LFW source.

**Review date:** 2026-10-24\
**Baseline repository commit:** `a92dc8025db1fab4d6e494804a876e58e4dde3cd`\
**Disposition:** C-level numerical regression passed after fixes; hardware and
dataset-accuracy acceptance remain open.

## Executive assessment

The committed golden traces are consistent with an independently implemented
integer fixed-point model. The actual HLS C++ also matches that model for the
tested block and connected-network cases. This verifies the golden-vector
contract, not every possible behavior of the unavailable original Python
training and quantization implementation.

Two implementation issues were corrected: fc2's undefined output byte strobes,
and missing per-image reset controls for the three spiking layers. A separate
numerical limitation remains: saturating intermediate accumulation is not
equivalent to a general Python matrix multiply under large input currents.
That limitation must be reconciled with the original Python specification
before declaring universal Python equivalence.

## Test method and evidence

The C++ bridge compiles the repository implementations against real Xilinx
Vitis HLS 2022.2 headers using g++. The independent Python specification uses
integer codes and explicit signed saturation and nearest-even rounding.
It does not call the C++ implementation to compute its expected outputs.

The suite checks all six individual stages, both available interface variants,
and a connected AXI chain alongside a connected plain-array chain. It compares
every output spike/pooled value, plain-array membrane state, active-tap count,
fc2 raw score word, TLAST, TKEEP and TSTRB. Exact integer equality is required.

| Campaign | Result |
|---|---|
| Pre-fix differential run | 2,632 checks; 3,533,117 values; 378 failing checks, all TSTRB |
| Post-fix differential run, seed 20261024, 32 random cases | 8,651 checks; 18,254,657 values; zero mismatches |
| Original golden C++ testbenches | All 12 pass; see `original_results.json` |
| Mutation sensitivity | See `mutation_results.json`; each deliberate defect must fail |

The post-fix campaign includes:

- Twenty golden timesteps at every stage and through the complete network.
- 327,680 LIF combinations spanning every membrane and beta code, five
  representative threshold codes, and boundary/tie currents.
- All 1,048,576 signed 20-bit current codes at zero initial membrane and fixed
  threshold, testing quantization, ties, saturation and firing boundaries.
- All 16 binary 2×2 pooling patterns for both pool blocks.
- Thirty-two seeded randomized weight sets covering zero, dense, sparse and
  random activation frames; signed weight extremes are included in generation.
- Nonstandard neuron parameter stress, including negative beta and threshold,
  to expose arithmetic defects beyond the supported deployment parameters.
- Sign-changing fc1 prefixes that overflow before the final sum, and 160-step
  positive/negative fc2 accumulation with final reset.
- Independent layer resets and image replay after intervening stateful input.

`results.json` records each comparison, source SHA-256 hashes, seed and run time.
`baseline.json` preserves the pre-fix evidence. Original testbench duplication
means 12 passing files are not 12 independent algorithms; the new suite directly
tests the plain-array conv1-pool and conv2 implementations missing from those
duplicated tests.

## Findings and changes

### F1 — fc2 leaves TSTRB undefined: corrected

`fc2_layer_axi.cpp` assigned data, keep and last, but not strb. C simulation
produced zero strobes in all score beats across 378 tested baseline frames. The updated
source explicitly sets all four strobe bits. Existing testbenches checked
TLAST but did not catch this. A zero strobe can be treated as a position byte
rather than data by stream consumers; the precise impact on the current DMA
requires RTL/board verification.

### F2 — membrane carryover between images: corrected in HLS source

conv1, conv2 and fc1 now expose `clear_state` on AXI-Lite. When asserted, every
neuron uses zero initial membrane for that invocation, then computes the first
timestep normally. The reset does not discard that timestep. Assert all three
flags together with fc2 `clear_accum` once per image. Replay tests cover resetting
after intervening inputs. Updated IP exports and new bitstream/handoff are
required; the committed XSA has no new reset registers.

### F3 — intermediate accumulator saturation: unresolved specification decision

The shared `acc_t` has range −128 through 127.999755859375. fc1 can sum 1,024
values with magnitude near one. With all inputs 1.0, 600 weights of 127/128
followed by 424 weights of −1.0 yield an unbounded dot product of **171.3125**.
The actual HLS sequential accumulator saturates on the positive prefix and then
on the negative suffix; at default parameters its membrane is −4.0 and it emits
**no spike**. A Python matmul followed by membrane clipping emits a spike.

Both behaviors are deterministic. The stress reference intentionally reproduces
the committed HLS semantics, and the golden traces do not trigger a discrepancy.
Widening accumulators is a candidate correction if the original Python uses
unbounded/float dot products, but doing so without that reference would change
the algorithm contract. This report does not hide the discrepancy behind a
passing test. fc2 can also saturate cumulative scores over time, creating ties
and potentially changing class ranking. The application exposes final score
saturation; full evaluation should quantify intermediate saturation as well.

### F4 — original Python and accuracy evidence: unavailable

The repository does not contain the original training code, quantization sweep,
model checkpoint, dataset/split, or deployment test arrays. Golden headers
provide weights and one trace fixture, not a held-out accuracy study. The
historical 93.75% software result was not independently reproduced here.

For the documented 80-image split, accuracy strictly above 96% requires at
least **77/80 = 96.25%**. Do not tune against those held-out images. Recover the
training pipeline, use a distinct validation split for tuning, lock the model,
and then evaluate the unchanged test split with paired fixed spike trains.
Multiple encoding seeds and a confidence interval should accompany the result.
No 96% accuracy claim is currently supported by this work.

### F5 — PL fc1 address map: still blocks board deployment

The committed XSA maps fc1 to a 64 KiB window although its weights begin at
offset 0x10000 and require another 64 KiB. Correct its range and nonoverlapping
base in the final PL phase. PS register offsets must be taken from regenerated
HLS headers/HWH after adding reset inputs, not assumed from the old XSA.

## Acceptance limits and next gates

Functional C++/Python tests do not prove finite-FIFO scheduling, RTL reset,
AXI-Lite reachability, DMA coherency, board initialization or physical timing.
The original Python reference must still resolve F3, and a locked held-out
dataset must establish accuracy. RTL co-simulation should test backpressure,
packet boundaries and consecutive images before final hardware acceptance.

The committed PL runs at 50 MHz; HLS targets 10 ns. WNS is timing margin, not
latency: negative WNS is a failed timing constraint. Until the requested 4 ns
constraint is clarified, the integration plan requires WNS ≥ 0 at the selected
clock. This report does not claim any achieved post-route WNS.
