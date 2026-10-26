# New LFW HLS architecture — standalone RTL cosimulation package

Copied on 2026-10-26 from the sibling `Spikeface_HLS` and repository verification assets. Originals were retained. The three production tops share `lfw_model.h`, `lfw_model.cpp`, and `lfw_axi.cpp`; they are not three separate source files. All design sources and testbenches are byte-identical copies. Only build/runner paths and misleading runner image-count metadata were adapted. `COPY_MANIFEST.json` records original and copied SHA-256 hashes.

## Run

Requires native Linux Vitis HLS 2022.2 / XSIM and its normal compiler/tool environment. Python 3 is needed only for the campaign runner. No PyTorch, dataset download, fixture regeneration, or sibling repository is required. The bundled fixture contains parameters, one image, and independent scalar-reference outputs for two timesteps.

From this folder, run a baseline campaign for all three blocks:

```bash
python3 stress_sim/run_rtl_cosim.py --name baseline_01 --jobs 3 --no-stalls
```

The runner locates `vitis_hls` on PATH, otherwise uses the original local installation. Set `VITIS_HLS` to override. Use a fresh campaign name each time. Results and logs go under this folder's `stress_sim/build/<name>`; summary JSON goes under `stress_sim`. Convolutions can take a long time; use `--jobs 1` to limit concurrent resource usage.

Run one block, random stalls, or the verification-only integrated finite-FIFO chain:

```bash
python3 stress_sim/run_rtl_cosim.py --name conv1_01 --jobs 1 --top lfw_conv1_axi --no-stalls
python3 stress_sim/run_rtl_cosim.py --name stalls_01 --jobs 3
python3 stress_sim/run_rtl_cosim.py --name chain_01 --jobs 1 --top lfw_chain_verify --no-stalls
```

Random stalls are enabled unless `--no-stalls` is supplied. A prior readout random-stall run failed inside XSIM; see evidence. The stress test replays the SAME image twice, clearing before each replay, with two timesteps per replay (four transactions). It checks every convolution spike and bit-exact readout scores plus TLAST/TKEEP/TSTRB. It does not establish distinct-image reset coverage at RTL.

Direct HLS invocation (single block, same stress test):

```bash
SPIKEFACE_HLS_STAGE=cosim SPIKEFACE_HLS_BLOCK=lfw_conv1_axi SPIKEFACE_COSIM_TEST=stress /home/pranavsai/WORK/Vitis_HLS/2022.2/bin/vitis_hls -f build_lfw_hls.tcl
```

Replace the top with `lfw_conv2_axi` or `lfw_readout_axi`. This builds under local `hls_build` and resets the selected project. Omitting `SPIKEFACE_COSIM_TEST=stress` uses the original two-transaction tolerance-based testbench. Default target: `xc7z020clg400-1`, 20 ns, fabric floating-point add/sub, and `-ffp-contract=off`. No simulation was rerun while assembling this package.

## Readiness review

| Top | AXI / dimensions reviewed | Existing logical evidence | Existing completed RTL evidence |
| --- | --- | --- | --- |
| `lfw_conv1_axi` | float32 AXIS input: 4096; 8-bit AXIS output: 8192; DDR params: 224 float32 | Native C simulation/synthesis and scalar-reference checks passed | Full cosimulation pass pending |
| `lfw_conv2_axi` | 8-bit AXIS input: 8192; output: 4096; DDR params: 3248 float32 | Native C simulation/synthesis and scalar-reference checks passed | Full cosimulation pass pending |
| `lfw_readout_axi` | 8-bit AXIS input: 4096; 32-bit score packets: 53; DDR params: 217247 float32 | Native C simulation/synthesis and scalar-reference checks passed | Four-transaction baseline PASS; random-stall attempt failed in simulator |

All three declare AXIS input/output, `m_axi` parameter access with matching depths, and AXI-Lite parameter addresses, scalar controls, and return/start control. Stream counts connect correctly. Parameter offsets fit their declared arrays. Static state is cleared by `clear_state`; assert it for every block on the first timestep of each image and keep it low thereafter. Review confirms BN before max pooling, strict-threshold spikes, delayed subtractive hidden reset, and temporal-mean leaky readout. Existing scalar-reference/mutation evidence supports these semantics.

**Verdict: all three are prepared to run RTL cosimulation; all three are not yet RTL-verified.** The source/interface review found no additional prerequisite code change. Prior evidence supports the sequential float32 numerical contract, but strict PyTorch score equivalence fails on 157/554 images despite matching all 554 class predictions. This unresolved numerical acceptance issue remains; copying the design does not resolve it.

The internal convolution streams have no TLAST and use fixed transfer counts. Readout asserts TLAST only on score 53, with KEEP/STRB=15. This is suitable for the supplied HLS testbenches and internal fixed-length streams; external DMA integration must respect that framing. Start downstream stages before upstream producers when connecting finite FIFOs. Full chain/backpressure verification, routed timing, and board validation remain pending.

## Evidence and provenance

`evidence/` contains copied historical results and reports; paths inside those reports refer to original campaigns, not new runs here. The Oct 25 `RTL_RESUME_REPORT.md` and `rtl_readout_baseline_result.json` supersede the Oct 24 report's earlier lack of any completed RTL pass. The preserved historical readout JSON labels two replays as two images; the actual testbench uses one distinct image. The copied runner now reports that correctly. Earlier synthesis resource estimates may use different float operator bindings; they are historical, not fresh estimates for this package.

Validation performed during packaging: copy/hash equality for unchanged inputs, binary fixture dimensions and length, Python syntax, and Tcl dependency resolution for all three tops plus the optional chain. No native C simulation, synthesis, or RTL simulation was rerun. The existing results are evidence from previous sessions; no source-hash-bound complete three-block RTL pass exists.
