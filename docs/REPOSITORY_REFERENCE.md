# SpikeFace repository reference

This is the initial 80-file snapshot. Subsequent work recovered the incompatible
53-class LFW Python model and made it authoritative at the owner's request.
Current implementation, verification and measured accuracy are documented in
[the project README](spikeface/README.md) and
[the LFW report](spikeface/stress_sim/LFW_REPORT.md). Vitis/Vivado were later
discovered outside PATH and native HLS checks were performed; the statements
below describe the initial read-only review.

Reviewed on 2026-10-24. Repository: https://github.com/chidellapranav/spikeface.git

Clone: `Branch_Linux/spikeface`. Reviewed commit: `a92dc8025db1fab4d6e494804a876e58e4dde3cd` (80 tracked files).
The parent checkout is older and has pre-existing changes; it was preserved.

## Architecture

SpikeFace implements a fixed-point spiking face-classification accelerator for PYNQ-Z2 / Zynq XC7Z020. The ARM processing system configures weights and neuron parameters, generates Bernoulli spike frames, starts each compute block per timestep, and transfers data through two AXI DMAs. The programmable logic computes:

`1024 binary pixels → conv1 (8×32×32) → pool1 (8×16×16) → conv2 (16×16×16) → pool2 (16×8×8) → fc1 (64 spikes) → fc2 (40 accumulated scores)`.

Convolutions use 5×5 kernels and padding 2. Pooling averages 2×2 spikes exactly into {0,.25,.5,.75,1}. Streams use channel-major then raster order. fc1 consumes the flattened pool2 output in that same order. Golden traces cover 20 timesteps; the deployment default is 50.

The LIF update computes beta×membrane + current, rounds and saturates to Q3.5, compares against threshold, and resets to zero on a spike. Default beta=.9375 and threshold=1.0. conv1/conv2/fc1 membranes persist across calls. fc2 has no LIF and accumulates weighted votes, resetting only when clear_accum is asserted. Prediction is argmax of the final 40 scores.

| Quantity | Representation |
|---|---|
| conv1/conv2/fc1 weights | signed ap_fixed<8,1>, 7 fractional bits |
| fc2 weights | signed ap_fixed<8,2>, 6 fractional bits |
| Spikes | ap_uint<1> |
| Pooled activations | ap_ufixed<3,1> |
| Membrane, beta, threshold | ap_fixed<8,3,AP_RND_CONV,AP_SAT> |
| Internal currents / readout accumulation | ap_fixed<20,8,AP_RND_CONV,AP_SAT> |
| DMA score payload | raw bits of ap_fixed<32,8,AP_RND_CONV,AP_SAT> |

Nonzero inputs gate MAC contributions. This establishes arithmetic sparsity; actual power savings require measurement. The loops still visit each scheduled tap. conv2 processes one input channel group at a time, and fc1 one input at a time. fc2 also processes one tap per cycle; older comments claiming a fully parallel fc2 are stale. Frame buffering means AXI streaming does not imply a fully overlapped pixel pipeline.

## Processing-system orchestration

`configure()` checks fixed-point grids and shapes, packs four weight bytes per little-endian word, writes AXI-Lite memories, and reads weights back. `run_inference()` accepts an image or fixed replay spikes. Per timestep it sets fc2 clear_accum only at t=0, starts fc2→fc1→conv2→conv1, arms output DMA, sends input DMA, and polls completion with a timeout. Both pool ap_start pins are tied high. Each timestep sends 1024 input bytes and receives 40 32-bit scores (160 bytes). fc2 asserts TLAST only on score 40. Optional results contain the score trace and active-tap sparsity.

`fresh_state=True` reloads the overlay and reconfigures parameters. This is the current documented workaround for membrane carryover and still needs validation on the board. Hardware bring-up should also verify DMA cache visibility, register-map field names, completion/error handling, and memory initialization.

## Hardware facts checked against the XSA

The archive CRC check passed. It contains the bitstream, system_top.hwh, PS initialization files, SmartConnect handoffs, and generated C drivers. Its main handoff records FCLK0=50 MHz. HLS IP configuration records a 10 ns target; this does not establish 100 MHz post-route operation.

| Block | PS base | Assigned span | Weight offset / data size |
|---|---|---|---|
| conv1 | 0x40000000 | 64 KiB | 0x100 / 200 bytes |
| conv2 | 0x40010000 | 64 KiB | 0x1000 / 3200 bytes |
| fc1 | 0x40020000 | 64 KiB | 0x10000 / 65536 bytes |
| fc2 | 0x40030000 | 64 KiB | 0x1000 / 2560 bytes |
| Input DMA | 0x41E00000 | 64 KiB | MM2S, 8-bit stream |
| Output DMA | 0x41E10000 | 64 KiB | S2MM, 32-bit stream |

**fc1 address mismatch:** weights need offsets 0x10000–0x1ffff, outside the assigned 64 KiB span. The driver detects this. The deployment guide proposes a 128 KiB range at a nonoverlapping aligned base such as 0x40040000, followed by hardware regeneration.

**State isolation:** conv1/conv2/fc1 have no explicit per-image clear_state. Clearing fc2 alone does not isolate images. Quantized leak can retain small nonzero membrane values indefinitely. A proper reset requires HLS/interface changes and rebuilt hardware.

## Build and verification boundaries

Six HLS exports are prerequisites for the Vivado restoration script: solution1/2/3/4 and solnfc1/2 under Spikeface_HLS, each with impl/ip. Generated exports are ignored by Git. README supplies a build-script example, but no standalone automated HLS build script is tracked. The Vivado Tcl recreates a project and block design; synthesis and implementation must be launched separately. Eclipse metadata contains original absolute paths and should not be treated as a portable build manifest.

Checks performed during this review:

- Read implementation and testbench source; inspected all tracked metadata and generated configurations.
- Parsed golden array initializers and compared all five adjacent stage boundaries: exact equality across 163840, 40960, 81920, 20480 and 1280 values respectively.
- Confirmed tb_conv1_pool.cpp equals tb_conv1_pool_axi.cpp byte for byte; both exercise the AXI pool. Likewise both tb_conv2_array files exercise the AXI conv2. Those pairs do not provide independent plain-array coverage.
- Python driver syntax compiled successfully without importing board dependencies.
- Parsed XML/JSON configuration and checked XSA archive integrity. The UI file uses Vivado layout syntax that is not strict JSON; this is not a runtime source issue.

C-simulation, RTL co-simulation, synthesis, timing closure, board inference, power, and dataset accuracy were not rerun. Vivado and Vitis HLS were absent from PATH. Documentation reports prior bit-exact C-simulation and 93.75% software accuracy; those are historical claims, not new measurements. The single-neuron test checks spikes but not its available golden membrane/current traces. fc2 test prints argmax comparisons but its failure count is driven by score and TLAST checks.

The Python training/reference implementation, quantization sweep, golden exporters, dataset split, deployment .npy assets, and referenced mock driver tests are absent from the tracked repository. Golden headers contain test weights and traces, but do not supply the complete reproducible training/evaluation workflow.

## Every tracked file

Paths below are relative to the clone. Large golden headers were inspected as structured data; binary hardware was inspected via its archive and handoff rather than interpreted as source.

| File | Purpose |
|---|---|
| `.gitignore` | Excludes generated HLS solutions and Vivado build products. |
| `PS/DEPLOYMENT_AND_ACCURACY.md` | Board deployment plan, known hardware blockers, reset workaround and evaluation procedure. |
| `PS/README.md` | PYNQ driver API, expected data formats and register map. |
| `PS/spikeface_driver.py` | Board runtime: weight encoding, MMIO, DMA scheduling, state reload, inference and diagnostics. |
| `README.md` | Main architecture, numeric design, verification claims and rebuild instructions. |
| `Spikeface_HLS/.apc/autopilot.apfmapping` | Vitis mapping of original fc2 source and testbench paths. |
| `Spikeface_HLS/.cproject` | Eclipse CDT compiler/build/indexer configuration. |
| `Spikeface_HLS/.project` | Eclipse project identity and linked resources with machine-specific absolute paths. |
| `Spikeface_HLS/.settings/Spikeface_HLS.Debug.launch` | Eclipse Debug launch configuration. |
| `Spikeface_HLS/.settings/Spikeface_HLS.Release.launch` | Eclipse Release launch configuration. |
| `Spikeface_HLS/.settings/language.settings.xml` | Eclipse language/indexer provider configuration. |
| `Spikeface_HLS/.vitis_hls_log_all.xml` | Historical Vitis HLS log dump; not an executable build recipe. |
| `Spikeface_HLS/README.md` | Historical fc1 checkpoint; its next steps and src/tb paths are stale. |
| `Spikeface_HLS/conv1_layer.cpp` | Include-only translation unit; conv1 implementation lives in conv1_layer.h. |
| `Spikeface_HLS/conv1_layer.h` | Plain-array conv1 implementation, dimensions, padding and persistent state supplied by caller. |
| `Spikeface_HLS/conv1_layer_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/conv1_layer_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/conv1_pool.cpp` | Plain-array layer/pooling implementation. |
| `Spikeface_HLS/conv1_pool.h` | Plain-array declarations, dimensions and configuration; conv2 also defines pooled MAC helpers. |
| `Spikeface_HLS/conv1_pool_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/conv1_pool_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/conv2_layer.cpp` | Plain-array layer/pooling implementation. |
| `Spikeface_HLS/conv2_layer.h` | Plain-array declarations, dimensions and configuration; conv2 also defines pooled MAC helpers. |
| `Spikeface_HLS/conv2_layer_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/conv2_layer_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/conv2_pool.cpp` | Plain-array layer/pooling implementation. |
| `Spikeface_HLS/conv2_pool.h` | Plain-array declarations, dimensions and configuration; conv2 also defines pooled MAC helpers. |
| `Spikeface_HLS/conv2_pool_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/conv2_pool_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/conv2_types.h` | Exact quarter-step pooled activation type. |
| `Spikeface_HLS/fc1_layer.cpp` | Plain-array layer/pooling implementation. |
| `Spikeface_HLS/fc1_layer.h` | Plain-array declarations, dimensions and configuration; conv2 also defines pooled MAC helpers. |
| `Spikeface_HLS/fc1_layer_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/fc1_layer_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/fc2_layer_axi.cpp` | AXI top implementation with stream interfaces and control pragmas. |
| `Spikeface_HLS/fc2_layer_axi.h` | AXI interface declaration and dimensions/types for the matching top. |
| `Spikeface_HLS/golden_conv1_array.h` | conv1 kernels, binary frames, spikes and active-tap counts. |
| `Spikeface_HLS/golden_conv2.h` | conv2 weights, pooled inputs, spikes and tap counts. |
| `Spikeface_HLS/golden_fc1.h` | fc1 weights, pooled input vectors, spikes and tap counts. |
| `Spikeface_HLS/golden_fc2.h` | fc2 weights, input spikes and cumulative class-score traces. |
| `Spikeface_HLS/golden_pool.h` | pool1 input spikes and expected average-pooled outputs. |
| `Spikeface_HLS/golden_pool2.h` | pool2 input spikes and expected outputs. |
| `Spikeface_HLS/golden_vectors.h` | Single-neuron weights, patches, spikes, membrane and current traces. |
| `Spikeface_HLS/hls.app` | Legacy Vitis project manifest, active fc2 top and solution names; includes stale file paths. |
| `Spikeface_HLS/lif_neuron.h` | Unrolled 25-tap gated binary MAC and shared LIF update. |
| `Spikeface_HLS/spikeface_neuron_top.h` | Single-neuron step combining gated MAC and LIF, pipelined at requested II=1. |
| `Spikeface_HLS/spikeface_types.h` | Shared fixed-point types, convergent rounding and saturation choices. |
| `Spikeface_HLS/tb_conv1_array.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_conv1_array_axi.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_conv1_pool.cpp` | Golden-vector C-simulation testbench; exercises AXI pool1 (duplicate of the _axi test). |
| `Spikeface_HLS/tb_conv1_pool_axi.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_conv2_array.cpp` | Golden-vector C-simulation testbench; exercises AXI conv2 (duplicate of the _axi test). |
| `Spikeface_HLS/tb_conv2_array_axi.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_conv2_pool.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_conv2_pool_axi.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_fc1_layer.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_fc1_layer_axi.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `Spikeface_HLS/tb_fc2_layer_axi.cpp` | Golden-vector C-simulation testbench; checks score values and TLAST framing. |
| `Spikeface_HLS/tb_spikeface_neuron.cpp` | Golden-vector C-simulation testbench; checks the corresponding neuron/layer output trace. |
| `spikeface/rebuild_vivado.tcl` | Vivado restoration, IP catalog paths, block wiring, addresses and run/report configuration. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_auto_pc_0/system_top_auto_pc_0.xci` | Generated IP configuration for xilinx.com:ip:axi_protocol_converter:2.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_axi_dma_0_0/system_top_axi_dma_0_0.xci` | Generated IP configuration for xilinx.com:ip:axi_dma:7.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_axi_dma_1_0/system_top_axi_dma_1_0.xci` | Generated IP configuration for xilinx.com:ip:axi_dma:7.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_conv1_layer_axi_0_0/system_top_conv1_layer_axi_0_0.xci` | Generated IP configuration for xilinx.com:hls:conv1_layer_axi:1.0. Recorded HLS latency: 11210 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_conv1_pool_axi_0_1/system_top_conv1_pool_axi_0_1.xci` | Generated IP configuration for xilinx.com:hls:conv1_pool_axi:1.0. Recorded HLS latency: 10253 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_conv2_layer_axi_0_0/system_top_conv2_layer_axi_0_0.xci` | Generated IP configuration for xilinx.com:hls:conv2_layer_axi:1.0. Recorded HLS latency: 1032797 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_conv2_pool_axi_0_0/system_top_conv2_pool_axi_0_0.xci` | Generated IP configuration for xilinx.com:hls:conv2_pool_axi:1.0. Recorded HLS latency: 5132 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_fc1_layer_axi_0_1/system_top_fc1_layer_axi_0_1.xci` | Generated IP configuration for xilinx.com:hls:fc1_layer_axi:1.0. Recorded HLS latency: 67462 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_fc2_layer_axi_0_2/system_top_fc2_layer_axi_0_2.xci` | Generated IP configuration for xilinx.com:hls:fc2_layer_axi:1.0. Recorded HLS latency: 2637 cycles. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_processing_system7_0_0/system_top_processing_system7_0_0.xci` | Generated IP configuration for xilinx.com:ip:processing_system7:5.5. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_ps7_0_axi_periph_0/system_top_ps7_0_axi_periph_0.xci` | Generated IP configuration for xilinx.com:ip:axi_interconnect:2.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_rst_ps7_0_50M_0/system_top_rst_ps7_0_50M_0.xci` | Generated IP configuration for xilinx.com:ip:proc_sys_reset:5.0. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_smartconnect_0_0/system_top_smartconnect_0_0.xci` | Generated IP configuration for xilinx.com:ip:smartconnect:1.0. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_smartconnect_1_0/system_top_smartconnect_1_0.xci` | Generated IP configuration for xilinx.com:ip:smartconnect:1.0. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_xbar_0/system_top_xbar_0.xci` | Generated IP configuration for xilinx.com:ip:axi_crossbar:2.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ip/system_top_xlconstant_0_0/system_top_xlconstant_0_0.xci` | Generated IP configuration for xilinx.com:ip:xlconstant:1.1. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/system_top.bd` | Native block design: six compute/pool stages, PS7, DMA, interconnects, clock/reset and address map. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/system_top.bda` | Generated GraphML block-design metadata. |
| `spikeface/spikeface.srcs/sources_1/bd/system_top/ui/bd_9edf8fa5.ui` | Vivado block-diagram graphical layout; no inference computation. |
| `spikeface/system_150926.xsa` | Binary hardware export: bitstream, hardware handoff, PS initialization and generated C drivers. |
