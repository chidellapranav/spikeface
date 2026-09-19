# SpikeFace — fc1 (Checkpoint 3 continued)

## What's different about fc1

Structurally different from conv1/conv2: no spatial patches, no padding,
no kernels. `fc1` is a dense/fully-connected layer — every one of the
1,024 pooled inputs (16 channels × 8×8, from `conv2_pool_axi`'s output,
already in the correct flat channel-major/row-major order) connects to
every one of 64 output neurons.

## Applying the conv2 lesson from the start

conv2's fan-in (200 taps/neuron) blew the LUT/DSP budget when fully
parallelized (221% of the chip). fc1's fan-in is **5x larger** (1,024
taps/neuron) — full parallelism here would be drastically worse. Rather
than rediscover that the hard way, `FC1_GROUP` (in `fc1_layer.h`) starts
directly at `1` — the verified-safe, time-multiplexed setting from conv2 —
processing one input at a time, looping sequentially over all 1,024.

**Note on expected latency**: despite the larger per-neuron fan-in, fc1
should end up *faster overall* than conv2, because it has far fewer output
neurons (64 vs. conv2's 4,096). Rough expectation: 64 × ~1,030 cycles/neuron
≈ 66,000 cycles/timestep — well under conv2's ~1.03M cycles/timestep. Real
synthesis numbers will confirm this.

## Files
- `src/fc1_layer.h` + `fc1_layer.cpp` — plain-array version.
- `src/fc1_layer_axi.h` + `fc1_layer_axi.cpp` — AXI-ready version (stream
  I/O, AXI-Lite control, `mem_state` made internal/`static`).
- `tb/tb_fc1_layer.cpp` + `tb/tb_fc1_layer_axi.cpp` + `tb/golden_fc1.h` —
  testbenches and golden data (golden vectors recomputed from scratch
  through the FULL chain: conv1→pool1→conv2→pool2→fc1, keeping weight
  normalization consistent across all layers).

## Verified results (both versions, first attempt, no bugs)
- Plain-array: **0/1,280 mismatches** (64 neurons × 20 timesteps).
- AXI-ready: **0/1,280 mismatches**, identical.
- Sparsity: **66.3%** of possible MAC taps skipped — the highest of any
  layer so far, consistent with activations getting sparser deeper into
  the network.

## Next steps
1. Run both through real Vitis HLS: csim → csynth for `fc1_layer_step`
   first (sanity check, matching the conv2 pattern), then the full ladder
   (csim → csynth → cosim → Package IP) for `fc1_layer_axi`.
2. Watch for timing/resource numbers — with `FC1_GROUP=1` this should be
   comfortably within budget, but confirm before considering any grouping
   increase (and if you do try one, learn from conv2: check *timing slack*
   as carefully as resource counts, since higher grouping costs
   combinational depth, not just area).
3. Then: fc2 (the readout layer — simpler structurally, no LIF, just an
   accumulator, small fan-in of 64) — the last HLS block needed before
   Vivado integration.
