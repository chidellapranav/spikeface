# SpikeFace — dataset inference and HLS verification

## Folders and entry points

Paths below are relative to this Git repository (`spikeface/`), not its parent
workspace. The project targets spiking face classification on PYNQ-Z2.

| Folder | Contents and role |
| --- | --- |
| [`newarch_HLS/`](newarch_HLS/README.md) | Separate optimized LFW implementation, HLS build/GUI scripts, bundled fixture, RTL campaign runner, and historical evidence. Recommended starting point for the proposed hardware automation. |
| [`Spikeface_HLS/`](Spikeface_HLS/README.md) | LFW baseline C++ sources and testbench used by the root HLS build and verification tools. Changes here do not automatically update `newarch_HLS/`. |
| [`PL_Design/`](PL_Design/README.md) | Current Vivado project, block design, hardware exports, and `create_lfw_pl.tcl` project/bitstream generator. Board presets are a separate prerequisite. |
| `hls_ip/` | Packaged conv1, conv2, and readout IP archives. Unpack the three per-IP `export.zip` files before Vivado registration. These do not update automatically when sources change. |
| [`PS/`](PS/README.md) | Current LFW board driver, host model/data bundle exporter, golden replay, and board evaluation CLI. |
| [`bootrun/`](bootrun/README.md) | Boot/deployment instructions, supplied `.bit`/`.hwh`/`.xsa` files, and runtime script copies. Refresh hardware and runtime copies together after changes. |
| `app/` | Python LFW evaluation and readout-refit utilities, plus legacy inference, demo, and HTTP commands. Entry point: `python -m app`. |
| [`stress_sim/`](stress_sim/README.md) | Root LFW references, fixture export, C++/AXI verification, mutation tests, RTL runners, and reports. Root tools generally exercise the baseline sources. |
| `tests/` | Python application, HTTP, PS driver, and bundle regression tests. |
| [`models/`](models/README.md) | Readout-refit candidate checkpoint and experiment report, separate from the original checkpoint. |
| `docs/` | PS/PL integration plan and historical repository review. Read dated status statements in their historical context. |
| [`Spike_old/`](Spike_old/README.md) | Archived 32x32, 40-class fixed-point HLS, Vivado, PS, and verification flow. Incompatible with the current LFW model. |
| `.git/` | Git metadata. |

Root files: `build_lfw_hls.tcl` builds the baseline; `requirements.txt` lists
NumPy/Pillow dependencies; `requirements-lfw.txt` adds PyTorch, snnTorch, and
SciPy; `.gitignore` excludes local environments and generated outputs.

## Hardware flow

```text
PS image -> input DMA -> lfw_conv1_axi -> lfw_conv2_axi
         -> lfw_readout_axi -> output DMA -> PS class scores
PS DDR parameters ----------> each HLS IP through AXI memory access
PS AXI-Lite ----------------> configuration, start, and state clear
```

The three production tops share `lfw_model.cpp`, `lfw_model.h`, and
`lfw_axi.cpp`. The default model repeats a normalized 64x64 grayscale image for
25 timesteps. Each timestep transfers 4096 float32 pixels in and 53 float32
scores out. Clear all blocks on the first timestep of each image and start
downstream blocks before upstream producers. Existing scripts target
`xc7z020clg400-1`, a 20 ns HLS clock, and 50 MHz PL operation using
Vitis HLS/Vivado 2022.2.

## CMake configuration for HLS and Vivado

Requires CMake 3.24+, Ninja for the supplied preset, and installed AMD tools.
Run from this repository root in a shell with the vendor tool environment set.
The current targets use the optimized `newarch_HLS/` sources and bundled fixture,
PYNQ-Z2, 20 ns, and fabric floating-point add/sub. The baseline root Tcl flow
remains available separately.

Configure with actual executable paths (replace these example paths):

```powershell
cmake --preset hardware `
  -DSPIKEFACE_HLS_EXECUTABLE="$env:SPIKEFACE_HLS_EXECUTABLE" `
  -DSPIKEFACE_VIVADO_EXECUTABLE="$env:SPIKEFACE_VIVADO_EXECUTABLE" `
  -DSPIKEFACE_BOARD_REPO="$env:SPIKEFACE_BOARD_REPO"
```

Alternatively set `SPIKEFACE_HLS_ROOT` and `SPIKEFACE_VIVADO_ROOT` to installation
folders containing `bin/`; tools are also searched on PATH. Linux example:

```bash
cmake --preset hardware \
  -DSPIKEFACE_HLS_ROOT="$SPIKEFACE_HLS_ROOT" \
  -DSPIKEFACE_VIVADO_ROOT="$SPIKEFACE_VIVADO_ROOT" \
  -DSPIKEFACE_BOARD_REPO="$SPIKEFACE_BOARD_REPO"
```

The board repository must contain `pynq-z2/A.0/board.xml`. Store local settings
in ignored `CMakeUserPresets.json` if desired. Without Ninja, use a supported
installed generator explicitly, for example `cmake -S . -B build/hardware -G
"Unix Makefiles"` on Linux with the same cache settings.

| Configuration | Default / meaning |
| --- | --- |
| `SPIKEFACE_HLS_EXECUTABLE`, `SPIKEFACE_VIVADO_EXECUTABLE` | Full launcher paths; take priority over discovery. |
| `SPIKEFACE_HLS_ROOT`, `SPIKEFACE_VIVADO_ROOT` | Installation roots used as discovery hints. |
| `SPIKEFACE_HLS_VERSION`, `SPIKEFACE_VIVADO_VERSION` | Expected reported release, both `2022.2`; an empty value skips the release comparison. |
| `SPIKEFACE_ENABLE_HLS`, `SPIKEFACE_ENABLE_VIVADO` | Both `ON`; disable either to configure only the other tool. |
| `SPIKEFACE_BOARD_REPO` | Local PYNQ-Z2 board repository. Checked before Vivado builds. |
| `SPIKEFACE_CONV1_IP`, `SPIKEFACE_CONV2_IP`, `SPIKEFACE_READOUT_IP` | Unpacked repositories containing `component.xml`; default to this build's HLS exports. |

Changing releases requires updating both the executable/root and expected
version. Use a new build directory when switching installations to avoid cached
paths and mixed IP. Version checks do not establish compatibility. The current
HLS launcher is `vitis_hls -f`; unified-flow releases may require an adapter
using `vitis-run --mode hls --tcl`, as described in [AMD documentation](https://docs.amd.com/r/en-US/ug1553-vitis-ide/vitis-run-Command).
Windows launchers are accepted, but native hardware execution on Windows has
not been validated; the existing RTL campaign documentation assumes Linux.

Build targets explicitly; the default build does not launch vendor tools:

```bash
cmake --build build/hardware --target hls-csim
cmake --build build/hardware --target hls-synth
cmake --build build/hardware --target hls-cosim
cmake --build build/hardware --target hls-export
cmake --build build/hardware --target vivado-bd
cmake --build build/hardware --target vivado-bit
```

Each HLS target processes all three production tops. It runs its complete
pipeline from C simulation in a separate `build/hardware/hls/<stage>/` directory;
`hls-export` additionally requires RTL co-simulation before packaging. These
explicit targets rerun when requested; they do not provide incremental stage
reuse. Do not run the same target concurrently from separate build processes.

Run `hls-export` before either Vivado target when using default IP paths.
Vivado targets do not automatically invoke HLS, so existing verified exports
can be reused. To use committed IP archives instead, unpack the three per-IP
ZIPs and configure `SPIKEFACE_CONV1_IP`, `SPIKEFACE_CONV2_IP`, and
`SPIKEFACE_READOUT_IP` explicitly. These must correspond to the intended sources.

`vivado-bd` creates and validates a project; `vivado-bit` independently creates
a project and runs implementation, timing/DRC checks, bitstream generation,
and matching `.bit`/`.hwh`/`.xsa` export. Each invocation writes to a fresh,
printed directory under `build/hardware/vivado/<mode>/`, preserving previous
runs. Board replay remains a separate deployment check.

Implementation files are `CMakeLists.txt`, `CMakePresets.json`,
`cmake/FindXilinxTools.cmake`, and `cmake/RunVivado.cmake`. CMake invokes the
existing vendor Tcl scripts; it does not install tools or board presets.

## Current model

The current target is the **64×64, 53-identity LFW model** supplied in
[`trmp_spike`](https://github.com/chidellapranav/trmp_spike), selected by the
project owner as the authoritative algorithm. The original 32×32 / 40-class HLS
and XSA are retained as a **legacy design**; their sources and documentation are in [`Spike_old`](Spike_old/README.md).
They are architecturally incompatible with the LFW checkpoint.

## Current measured results

| Experiment | Correct / images | Accuracy |
|---|---|---|
| Supplied LFW PyTorch checkpoint | 406 / 554 | **73.2852%** |
| New LFW HLS C++ and AXI simulation, same checkpoint | 406 / 554 | **73.2852%** |
| Validation-selected readout refit, PyTorch | 427 / 554 | **77.0758%** |

The new HLS makes the same predicted class as PyTorch on all 554 supplied test
images. It is **not bit-exact to PyTorch**: 157 images fail the strict score
tolerance, and recurrent threshold crossings amplify small floating-point
differences. The independent scalar float32 reference is bit-exact in the tested
cases. The strict suite intentionally retains and reports these counterexamples.
See [the LFW verification report](stress_sim/LFW_REPORT.md), not only a PASS line.

**The >96% target is not achieved.** On 554 images it requires at least 532
correct. The supplied training used its test split for checkpoint selection, so
these are development results rather than a new untouched-holdout claim.
No board-accuracy or routed-WNS result is claimed.

## Run accuracy inference on a dataset

Python 3.10+, NumPy, Pillow, CPU PyTorch and snnTorch are required for LFW.
From this repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install snntorch==0.9.4 scipy
git clone https://github.com/chidellapranav/trmp_spike.git ../trmp_spike
python -m app lfw-evaluate --repo ../trmp_spike --output accuracy.json
```

If you have already cloned the reference repo and installed PyTorch /
snnTorch in the ignored verification build folder, you can reuse that
installation:

```bash
PYTHONPATH=stress_sim/build/python_deps python3 -m app lfw-evaluate \
  --repo ../trmp_spike --output accuracy.json
```

The output JSON includes accuracy, confidence interval, per-class counts,
confusion matrix, every prediction and score vector, and model/data hashes.
`--limit N` is a clearly marked partial health check, not full-dataset accuracy.

Select `--backend hls` to measure actual C++/AXI simulation accuracy on the same
dataset (requires the Xilinx headers and g++). The default `--backend software`
runs the original PyTorch model. `--backend pynq --bitstream /path/model.bit`
connects the future new LFW hardware; it cannot use the committed legacy XSA.

For a different array database, use `--dataset-dir /path/to/dataset`. It must
contain `X_test.npy` with shape `(N,64,64)` and finite float pixels in `[0,1]`,
`y_test.npy` with integer class indices, and `class_names.npy` matching the exact
53-class checkpoint label ordering. A checkpoint cannot recognize arbitrary new
identities without retraining its classifier.

For labeled image folders:

```text
my_faces/
  Tony Blair/
    face_001.png
  George W Bush/
    face_002.png
```

```bash
python -m app lfw-evaluate --repo ../trmp_spike \
  --images-root /path/to/my_faces --output my_accuracy.json
```

Folder names must match the reference `class_names.npy`. Images must already be
aligned and resized to 64×64 using the training preprocessing. The app refuses
to silently resize arbitrary faces using a different interpolation/alignment
pipeline and then present that as comparable accuracy. Input encoding repeats
the same grayscale image for all 25 timesteps; it is not Bernoulli rate coding.

## Readout accuracy experiment

The hidden layers and architecture are unchanged. Regularization was selected
using a training-only stratified validation split, then refitted on all training
images. The test set was evaluated once after selection. The original frozen
checkpoint's previous test-based selection remains a caveat.

```bash
python -m app lfw-evaluate --repo ../trmp_spike \
  --checkpoint models/lfw_readout_refit.pt --output candidate_accuracy.json
```

Reproduce the refit with `python -m app.lfw_refit --repo ../trmp_spike
--output-dir /path/to/output` (also requires SciPy). See
[models/README.md](models/README.md). This candidate is not substituted silently
for the original model or used to claim a 96% result.

## Verify the hardware algorithm

```bash
export XILINX_HLS_INCLUDE=/path/to/Vitis_HLS/2022.2/include
python Spike_old/stress_sim/run.py --cases 32
python Spike_old/stress_sim/original_tests.py
python Spike_old/stress_sim/mutations.py
python stress_sim/lfw_verify.py --repo ../trmp_spike
python stress_sim/lfw_mutations.py --repo ../trmp_spike
python stress_sim/lfw_dataset.py --repo ../trmp_spike
```

The first three commands test the legacy fixed-point pipeline. The last three
test the authoritative LFW architecture and require PyTorch/snnTorch. The strict
LFW verification and dataset commands currently **exit nonzero** for recorded
PyTorch numerical divergence, even though scalar-reference and classification
agreement checks pass. This is an intentional release gate, not a reason to
remove or relax a failing test. Reproduce the PyTorch dataset baseline first
before running `lfw_dataset.py`.

Legacy regression: **8,651 checks / 18,254,657 values**, all 12 original
testbenches pass, and four injected defects are detected. New LFW tests compare
every intermediate stage, resets, strict threshold boundaries, score bits and
packet fields; four LFW injected defects are detected. Full datasets and source
hashes are recorded in `stress_sim/*results.json` and `*evaluation.json`.

The native HLS flow is separate:

```bash
python stress_sim/export_lfw_fixture.py --repo ../trmp_spike
SPIKEFACE_HLS_STAGE=synth vitis_hls -f build_lfw_hls.tcl
# Optional generated-Verilog check (can be slow):
SPIKEFACE_HLS_STAGE=cosim vitis_hls -f build_lfw_hls.tcl
```

The LFW clock defaults to 20 ns / 50 MHz; `SPIKEFACE_CLOCK_NS` overrides it.
Native C-simulation and synthesis succeeded for all three new tops. HLS timing
estimates are not routed WNS. Final PL acceptance requires **WNS ≥ 0 ns**, as
requested. Integration remains gated on numerical-contract acceptance and
hardware verification. [PS and PL plan](docs/PS_PL_PLAN.md) describes DDR
parameters, three new IPs, reset scheduling and board evaluation.

## Optional legacy tools

`python -m app demo` replays the old golden fixture. `python -m app serve` runs
an optional loopback browser interface for the **legacy 40-class** model; it
does not implement LFW inference. `app infer`, `app evaluate` and
`app export-fixture` likewise use the explicitly documented legacy model format.
Their PYNQ backend requires compatible legacy hardware with the address-map fix.
The new `PS/lfw_driver.py` is a board-side candidate for the 53-class interface,
pending a matching new bitstream/HWH and real-board validation.

Application/PS tests: `python -m unittest discover -s tests -v`. HTTP tests open
a temporary loopback port. Neither browser visual QA nor physical board tests
are implied by unit tests.


## 5. Verify the LFW Prototype on PYNQ-Z2

This procedure applies to the **current 64x64, 53-identity LFW design** and
assumes its PL implementation and timing have already been verified. Use the
matching LFW hardware exported from your PL build, not the legacy 40-class XSA
under `Spike_old/`. The board verification has two stages: golden replay
to check hardware outputs, then test-set inference to measure accuracy.

### 5.1 Prepare the hardware and runtime

Boot the PYNQ-Z2 with its PYNQ image and use the board's PYNQ Python environment.
The board runner requires **NumPy and PYNQ**; PyTorch and snnTorch are needed only
on the host for export. Check that the hardware exposes `lfw_conv1_axi_0`,
`lfw_conv2_axi_0`, `lfw_readout_axi_0`, `axi_dma_0` and `axi_dma_1`, with register
offsets matching `PS/lfw_driver.py` and the exported HWH/driver headers.

Prepare matching `.bit` and `.hwh` files with the **same basename**, such as
`lfw_system.bit` and `lfw_system.hwh`. If only `.bit` and `.xsa` are available,
extract the HWH from that matching XSA. The runtime loads the bit/HWH pair.

### 5.2 Export the model and test data on the host

From the repository root, in the host environment containing the reference
model dependencies, run:

```bash
python -m PS.export_lfw --repo ../trmp_spike --output board_bundle
```

Adjust `--repo` to the reference checkout's location. Add `--checkpoint PATH`
only when intentionally selecting another compatible checkpoint. Use a fresh
output directory. The export packs the weights, biases, BN affine parameters,
model configuration, LFW test images and labels, and expected score traces for
two distinct test images. It also records checkpoint and bundle hashes.

Training and testing use separate image splits from LFW, covering the same 53
identities. This workflow evaluates **`X_test.npy` against `y_test.npy`**, using
the exact ordering in `class_names.npy`; it does not train on the board. The
original training used test accuracy to select its checkpoint, so this split
is not an untouched final holdout despite being separate from training images.

### 5.3 Copy the deployment files to the board

Transfer the following into a working folder such as
`/home/xilinx/spikeface/`:

```text
spikeface/
  PS/                      # Copy this repository's PS directory
  board_bundle/
    manifest.json
    lfw_bundle.npz
  lfw_system.bit
  lfw_system.hwh
```

Run the following board commands from that working folder. The training
checkout and PyTorch are not required on the board.

### 5.4 Run golden replay on the board

```bash
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --output smoke.json
```

Golden replay sends known images through the FPGA and compares its actual
53-class scores at **every timestep** with the exported sequential float32
reference. It runs **image A, image B, then image A again**, clearing network
state for each image. This checks reference agreement, repeatability and
different-image reset behavior. The default model repeats the same normalized
64x64 grayscale image for 25 timesteps; it does not use Bernoulli encoding.

Require `passed: true` in `smoke.json`. A mismatch, DMA error or timeout stops
the run with a nonzero exit. Resolve the failure before measuring accuracy.
The reference follows the HLS numerical contract; it is not a claim of bitwise
equivalence to PyTorch's reduction order. Passing these cases is a bring-up
check, not proof of correctness for every possible input.

### 5.5 Measure test-set accuracy on the board

First run a short check, then the complete test set:

```bash
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --mode evaluate --limit 5 --output five_images.json
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --mode evaluate --output board_accuracy.json
```

Both commands repeat golden replay before evaluation. For each test image,
the FPGA produces 53 scores; the largest score selects the predicted identity.
Compare that prediction with the true label in `y_test.npy`:

**Accuracy (%) = correct test predictions / total test images evaluated x 100.**

The full report records accuracy, correct/image counts, predictions, labels,
scores, confusion matrix, fabric clock and hardware/model hashes. Limited runs
are marked partial and must not be reported as full-dataset accuracy. Dataset
elapsed time includes Python orchestration, not just accelerator execution.
The original checkpoint's simulation baseline is **406/554 (73.2852%)**;
report the actual board result rather than assuming it matches that baseline.

### 5.6 Diagnose a failed run

Check IP names and register offsets, the loaded bit/HWH pairing, DDR parameter
connectivity, DMA stream widths and output TLAST. The driver checks transfer
sizes of **16384 input bytes and 212 output bytes per timestep**, waits for IP
completion and checks finite output scores. After a DMA error or timeout,
rerun the command to reload the overlay before retrying. `--timeout SECONDS`
adjusts the default 60-second per-wait timeout. Ensure no other process owns
the same DMA channels.

See [PS/LFW_BOARD_RUN.md](PS/LFW_BOARD_RUN.md) for the deployment instructions
and [PS/lfw_driver.py](PS/lfw_driver.py) for the board scheduler.
