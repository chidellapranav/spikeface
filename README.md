# SpikeFace — dataset inference and HLS verification

## Folders and entry points

Paths below are relative to this Git repository (`spikeface/`), not its parent
workspace. The project targets spiking face classification on PYNQ-Z2.

| Folder | Contents and role |
| --- | --- |
| [`newarch_HLS/`](newarch_HLS/README.md) | Separate optimized LFW implementation, HLS build/GUI scripts, bundled fixture, RTL campaign runner, and historical evidence. Source implementation used by the CMake HLS targets. |
| [`Spikeface_HLS/`](Spikeface_HLS/README.md) | LFW baseline C++ sources and testbench used by the root HLS build and verification tools. Changes here do not automatically update `newarch_HLS/`. |
| [`PL_Design/`](PL_Design/README.md) | Current Vivado project, block design, hardware exports, and `create_lfw_pl.tcl` project/bitstream generator. PYNQ-Z2 board presets are bundled under `board_files/`. |
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

## Build with CMake: start here

This walkthrough builds the current **64x64, 53-class LFW design for PYNQ-Z2**.
CMake coordinates Vitis HLS and Vivado; the vendor tools perform the hardware
work. You do not need a physical board to generate a Vivado project or bitstream.
You need a board only for deployment and inference.

The repository includes the HLS test fixture, packaged IP archives, Vivado Tcl,
and **PYNQ-Z2 board definitions in `PL_Design/board_files/pynq-z2/A.0/`**.
No separate board-file download or installation into Vivado is required.
The bundled files are `board.xml`, `part0_pins.xml`, and `preset.xml`; see
[board-file provenance](PL_Design/board_files/README.md).

### 1. Install and check the required software

| Software | Requirement |
| --- | --- |
| CMake | Version 3.24 or newer. |
| Ninja | Build launcher used by the commands below. |
| Vivado | 2022.2, with Zynq-7000 device support and any license required by your installation. |
| Vitis HLS | 2022.2 for rebuilding HLS IP. Not needed for the packaged-IP Vivado route. |
| Git | Optional if you download and extract the repository ZIP instead. |

On Windows, open **PowerShell**. If CMake or Ninja is missing, install them:

```powershell
winget install --id Kitware.CMake --exact
winget install --id Ninja-build.Ninja --exact
```

Close and reopen PowerShell after installation. Confirm both commands work:

```powershell
cmake --version
ninja --version
```

Install AMD tools separately before continuing. CMake does not install them.
Python, PyTorch, model downloads, and the external training repository are not
needed for the CMake hardware flow using the bundled fixture and packaged IPs.

### 2. Open your repository folder and identify the tool launchers

Clone the repository, or download its ZIP and extract it fully. In PowerShell,
use `Set-Location` followed by your actual extracted repository directory. This
must be the folder containing `CMakeLists.txt`, `README.md`, and `PL_Design`.
Then run:

```powershell
$repo = (Get-Location).Path
Test-Path "$repo/CMakeLists.txt"
Test-Path "$repo/PL_Design/board_files/pynq-z2/A.0/board.xml"
```

Both must return `True`. If not, open the correct directory or obtain the complete
updated repository. Paths in this guide are calculated from `$repo`, so the
checkout can be on any drive and its name can contain spaces.

Find your installed Vivado folder in File Explorer. Inside its `bin` directory,
locate `vivado.bat`. Do the same for `vitis_hls.bat` in the Vitis HLS installation
if you plan to rebuild the IPs. Save the full paths in local variables:

```powershell
$vivado = Read-Host 'Paste the full path to bin/vivado.bat, without surrounding quotes'
Test-Path -LiteralPath $vivado
& $vivado -version
```

`Test-Path` must return `True`; the version output must identify 2022.2.
These paths stay on your machine. They are not written into shared source files.
PowerShell does not have a `source` command; do not use Linux `source` syntax
or add `\d` before a path. The launcher can normally be invoked directly as above.

### 3. Choose a build route

| Route | Use it when | What it proves |
| --- | --- | --- |
| A: packaged IPs | You want to test Vivado without waiting for HLS synthesis. | The supplied IP archives integrate into the generated design. It does not rebuild or verify edited C++ sources. |
| B: rebuild HLS | You changed HLS sources or want newly generated hardware. | Executes the selected HLS verification/build stages before Vivado consumes their exports. |

Use separate build directories for the two routes. Generated files stay there,
not in the source folders. Keep the same PowerShell window open so the variables
above remain defined. If you reopen it, repeat step 2 and define the route's build
variable again.

### Route A: create the Vivado design using included IPs

Configure a new build directory:

```powershell
$build = Join-Path $repo 'build/packaged-vivado'
cmake -S "$repo" -B "$build" -G Ninja `
  "-DSPIKEFACE_VIVADO_EXECUTABLE=$vivado" `
  -DSPIKEFACE_ENABLE_HLS=OFF `
  -DSPIKEFACE_USE_PACKAGED_IP=ON
```

PowerShell continuation backticks must be the last character on each line, with
no trailing spaces. Successful configuration prints the Vivado path and release,
then `Configuring done`, `Generating done`, and the build directory.
CMake automatically unpacks all three committed IP archives into
`$build/packaged_ip/` and selects the bundled board files. No manual unzip or
board-path setting is needed.

Create and validate the block design:

```powershell
cmake --build "$build" --target vivado-bd
```

This creates the Zynq PS, DMA, three HLS blocks, interconnects, reset/clock wiring,
and address map, validates the design, generates output products, and exports a
hardware handoff without a bitstream. It does not run full design synthesis or
implementation. Output-product generation can still take time.

The terminal prints a fresh directory under `$build/vivado/bd/`. To inspect the
result, open `lfw_pynq_z2.xpr` from that directory in Vivado 2022.2, then open the
`lfw_system` block design. The command must finish successfully; merely seeing a
project file is not evidence that every build step passed.

### Route B: verify and rebuild all three HLS IPs

Set and check the HLS launcher, then configure a separate build:

```powershell
$hls = Read-Host 'Paste the full path to bin/vitis_hls.bat, without surrounding quotes'
Test-Path -LiteralPath $hls
& $hls -version

$build = Join-Path $repo 'build/rebuilt-hardware'
cmake -S "$repo" -B "$build" -G Ninja `
  "-DSPIKEFACE_HLS_EXECUTABLE=$hls" `
  "-DSPIKEFACE_VIVADO_EXECUTABLE=$vivado" `
  -DSPIKEFACE_USE_PACKAGED_IP=OFF
```

Check the configuration completes successfully. Run C simulation first:

```powershell
cmake --build "$build" --target hls-csim
```

Only continue after a successful exit. To inspect resource and timing estimates
before attempting RTL verification, optionally run:

```powershell
cmake --build "$build" --target hls-synth
```

To produce the IPs that Vivado will use, run:

```powershell
cmake --build "$build" --target hls-export
```

`hls-export` runs C simulation, synthesis, RTL co-simulation, and packaging for
all three production tops. RTL co-simulation can take a long time. Successful
synthesis alone does not mean export or RTL verification has completed.

These targets use the optimized `newarch_HLS/` sources, 20 ns, fabric floating-point
add/sub, the stress testbench, and no randomized stalls. Each target runs its
complete pipeline in an independent stage directory because the existing Tcl
resets HLS projects. Thus `hls-export` repeats work even after `hls-synth`.
`hls-cosim` is also available for RTL verification without export. Do not run the
same target concurrently from separate terminals.

Check that all exported IP definitions exist:

```powershell
Test-Path "$build/hls/export/lfw_conv1_axi/sol/impl/ip/component.xml"
Test-Path "$build/hls/export/lfw_conv2_axi/sol/impl/ip/component.xml"
Test-Path "$build/hls/export/lfw_readout_axi/sol/impl/ip/component.xml"
```

All three must return `True`, and `hls-export` must have exited successfully.
Now create the Vivado design from those new exports:

```powershell
cmake --build "$build" --target vivado-bd
```

Vivado targets do not automatically run HLS; this lets users reuse completed
exports without repeating a long HLS campaign.

### 4. Generate the FPGA bitstream (either route)

After your route succeeds, use its same `$build` directory:

```powershell
cmake --build "$build" --target vivado-bit
```

This creates a fresh project, runs synthesis and routing, checks setup/hold
slack and DRC errors, then generates the bitstream and matching hardware files.
It does not reuse the project created by `vivado-bd`. Timing failure stops the
flow; do not treat an incomplete run as deployable hardware.

Each invocation prints its unique output directory under `$build/vivado/bit/`.
On success, that directory contains:

| Output | Meaning |
| --- | --- |
| `lfw_system.bit` | FPGA programming image. |
| `lfw_system.hwh` | Matching hardware metadata used by PYNQ. |
| `lfw_system.xsa` | Hardware handoff including the bitstream. |
| `lfw_pynq_z2.xpr` | Vivado project for inspection. |
| `timing_summary.rpt`, `utilization.rpt`, `drc.rpt` | Routed timing, resources, and design-rule results. |

Always deploy the `.bit` and `.hwh` from the same successful run. Building hardware
does not measure face-recognition accuracy. Follow [the boot and deployment
guide](bootrun/README.md) and [board replay instructions](PS/LFW_BOARD_RUN.md)
for the additional model/data export and physical-board checks.

### 5. Build outside the repository, if preferred

The `$build` directory may be any new writable folder outside `PL_Design`,
including a folder on another drive. To select it interactively before a route's
configure command, use:

```powershell
$build = Read-Host 'Enter the full path of your chosen build folder'
```

Keep `-S "$repo"` as the source and `-B "$build"` as the output. Never use the
repository root itself as the build directory. Reuse the same output folder for
subsequent `cmake --build` commands. Choose a fresh one when switching between
packaged and rebuilt IPs or changing tool installations; CMake caches settings.

### 6. Configuration reference and version changes

| CMake setting | Default and purpose |
| --- | --- |
| `SPIKEFACE_HLS_EXECUTABLE`, `SPIKEFACE_VIVADO_EXECUTABLE` | Explicit installed launchers; otherwise discover tools on PATH. |
| `SPIKEFACE_HLS_ROOT`, `SPIKEFACE_VIVADO_ROOT` | Optional installation directories containing `bin`; discovery hints instead of explicit launcher paths. |
| `SPIKEFACE_HLS_VERSION`, `SPIKEFACE_VIVADO_VERSION` | Expected reported releases, both `2022.2`. |
| `SPIKEFACE_ENABLE_HLS`, `SPIKEFACE_ENABLE_VIVADO` | Both `ON`; disabling a tool removes its targets and discovery requirement. |
| `SPIKEFACE_USE_PACKAGED_IP` | `OFF`: use this build's HLS exports. `ON`: unpack committed archives for Vivado at configure time. |
| `SPIKEFACE_BOARD_REPO` | Defaults to this checkout's `PL_Design/board_files`; no override needed normally. |
| `SPIKEFACE_CONV1_IP`, `SPIKEFACE_CONV2_IP`, `SPIKEFACE_READOUT_IP` | Optional overrides for unpacked IP repositories containing `component.xml`. Explicit cached paths take priority over either IP mode. |

To change a setting, repeat the configure command with the new `-D` value, then
build. A plain `cmake --build "$build"` has no hardware target selected and does
not launch a hardware build. Select a target explicitly.

Version settings check the executable; they do not install or migrate tools.
For another release, select its actual launcher and expected version in a fresh
build directory. Only the existing `vitis_hls -f` interface is implemented.
Unified HLS releases may require changes for `vitis-run --mode hls --tcl`; see
[AMD's command-line reference](https://docs.amd.com/r/en-US/ug1553-vitis-ide/vitis-run-Command).
Passing a version check alone does not prove script or IP compatibility.

Store repeatable personal paths in ignored `CMakeUserPresets.json` if desired;
never commit local caches or personal absolute paths. The shared `hardware`
preset is an optional alternative to explicit `-S`/`-B` commands.

### 7. Linux equivalent

The detailed walkthrough above uses Windows PowerShell. On Linux, use Bash,
installed `cmake` and `ninja`, and the vendor environment appropriate to your
installation. Do not paste PowerShell backticks or `$env:` syntax into Bash.
From the repository root, a packaged-IP configuration is:

```bash
repo="$PWD"
read -r -p 'Full path to the Vivado executable: ' vivado
build="$repo/build/packaged-vivado"
cmake -S "$repo" -B "$build" -G Ninja \
  "-DSPIKEFACE_VIVADO_EXECUTABLE=$vivado" \
  -DSPIKEFACE_ENABLE_HLS=OFF \
  -DSPIKEFACE_USE_PACKAGED_IP=ON
cmake --build "$build" --target vivado-bd
```

For a full HLS rebuild, use a separate directory and configure both executables:

```bash
read -r -p 'Full path to the vitis_hls executable: ' hls
build="$repo/build/rebuilt-hardware"
cmake -S "$repo" -B "$build" -G Ninja \
  "-DSPIKEFACE_HLS_EXECUTABLE=$hls" \
  "-DSPIKEFACE_VIVADO_EXECUTABLE=$vivado" \
  -DSPIKEFACE_USE_PACKAGED_IP=OFF
cmake --build "$build" --target hls-csim
cmake --build "$build" --target hls-export
cmake --build "$build" --target vivado-bd
cmake --build "$build" --target vivado-bit
```

Run each build command only after the previous command succeeds.

### 8. Troubleshooting

| Message or symptom | Action |
| --- | --- |
| `cmake` or `ninja` is not recognized | Install the missing program, reopen the terminal, and check its version. |
| `vitis_hls not found` or `vivado not found` | Pass the actual launcher using the corresponding `SPIKEFACE_*_EXECUTABLE` setting; check it with `Test-Path`. |
| `expected 2022.2` | Select the 2022.2 executable, or intentionally configure another supported release and its expected version in a fresh build directory. |
| Launcher exists but cannot query version | Run `& $vivado -version` or `& $hls -version` directly. If needed, open the vendor command shell and configure there. |
| Need to load a Windows `.bat` environment | Open Command Prompt, run `call` followed by the quoted settings-file path, then run CMake in that same window. PowerShell commands/variables must be translated to cmd syntax. |
| Missing board preset | Check `PL_Design/board_files/pynq-z2/A.0/board.xml`. Use a complete updated checkout. An older cache may have an empty/custom board path; use a fresh build folder or explicitly set the bundled directory. |
| Missing `component.xml` | For route A enable packaged IPs in a fresh folder. For route B finish `hls-export`. Check for old explicit IP path overrides in the cache. |
| Build appears slow | HLS synthesis/co-simulation and Vivado output-product generation can be lengthy. Inspect the printed log/project directory; use route A to skip rebuilding HLS. |
| `vivado-bit` stops at timing or DRC | Inspect the reports in the printed run directory; resolve the hardware issue before deployment. |
| Need to inspect a previous Vivado run | Each invocation preserves its own timestamped directory. Open that run's `.xpr`; a new invocation does not overwrite it. |
| Configured the wrong folder or generator | Choose a fresh build directory and repeat configuration. Do not delete source or board files. |

The owner reported a successful Windows 2022.2 `vivado-bd` run using the supplied
IP archives and these board files. This does not establish complete HLS export,
bitstream generation, timing closure, or board inference for every installation.

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
