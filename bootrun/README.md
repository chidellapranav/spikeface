# Run SpikeFace on PYNQ-Z2

Follow sections 1–7 to boot the board and run the **64×64 / 53-class LFW**
accelerator using the supplied hardware export. Commands marked **host** run
on your Linux workstation; commands marked **board** run after SSH login.
This is SD-booted PYNQ Linux with an overlay loaded by Python after login.

> **Recommended:** clone this Git repository, open the project in **Vivado
> 2022.2**, and regenerate the bitstream and hardware export (`.bit` and `.xsa`,
> with **Include bitstream** enabled). Use the matching `.hwh` from that same
> build. Section 9 gives the regeneration steps and IP repository paths.
> This keeps the deployed hardware consistent with your checkout and IPs.

## Packaged files in this folder

```text
bootrun/
├── README.md
├── lfw_system.bit       # FPGA bitstream
├── lfw_system.hwh       # Matching top-level hardware metadata
├── lfw_system.xsa       # Copy of lfw_system_301026.xsa
└── PS/                  # Only the two board runtime scripts
    ├── lfw_driver.py
    └── run_lfw.py
```

All boot, deployment, and execution instructions are in this README. The
`PS/` subfolder preserves the import path used by `python3 -m PS.run_lfw`; it
contains only the board driver and runner. Host-side bundle export uses the
original `PS/export_lfw.py` at the repository root.

The PYNQ image is downloaded separately. **No model bundle currently exists in
this workspace**; generate `board_bundle/manifest.json` and
`board_bundle/lfw_bundle.npz` using section 4 before running inference. Hardware
exports and runtime copies alone do not include trained parameters/test data.
After regenerating hardware or editing the original PS code, refresh these
copies together before deployment. The XSA is retained for handoff; only the
matching `.bit`/`.hwh`, runtime and generated model bundle go to the board.

## 1. Files you need

Paths below are relative to the repository root. The new hardware archive has
been inspected: its ZIP integrity check passed and it includes the files listed
below. Board execution and accuracy still need to be measured.

| File / directory | Purpose |
| --- | --- |
| `PL_Design/lfw_pynq_z2_build/lfw_system_301026.xsa` | New hardware export, containing `lfw_system_301026.bit` and top-level `lfw_system.hwh` |
| `PL_Design/lfw_pynq_z2_build/lfw_pynq_z2.runs/impl_1/lfw_system_wrapper.bit` | Standalone generated bitstream; its SHA-256 matches the bitstream inside the new XSA. This generated run folder is Git-ignored; use the XSA extraction below after cloning |
| `PS/lfw_driver.py` | Board control, DDR parameter buffers, DMA, and state reset |
| `PS/run_lfw.py` | Golden replay and dataset evaluation CLI |
| `PS/export_lfw.py` | Host tool to create the model/data bundle |
| `board_bundle/manifest.json` | Generated model configuration, class ordering, timestep count, and hashes |
| `board_bundle/lfw_bundle.npz` | Generated float32 parameters, dataset, labels, and golden traces |
| PYNQ-Z2 SD image | Download separately using the link in section 2 |

The model bundle is generated in section 4; hardware files alone do not contain
trained runtime parameters or test data. The older `lfw_system.xsa` lacks a
bitstream; use **`lfw_system_301026.xsa`** for the supplied-export path.
The previous fixed-point hardware under `Spike_old/` is incompatible.

## 2. Download the image and boot PYNQ Linux

Download the **PYNQ-Z2 v3.1.1 SD-card image** separately:

[Download pynq_z2_v3.1.1.zip](https://download.amd.com/opendownload/pynq/pynq_z2_v3.1.1.zip)

The image is listed under **Previous downloads → PYNQ-Z2 → v3.1.1** on the
[official PYNQ website](https://www.pynq.io/). It is not included in this repository.

1. Extract the downloaded archive and write its image to a microSD card with
   an image-writing utility. Writing an image erases the selected device;
   check that you selected the intended SD card.
2. With the board powered off, insert the card and select **SD boot** using
   the board's labeled boot-mode jumper. Follow the board setup instructions
   for the jumper position and power-source selection.
3. Connect Ethernet to your host/network, connect the correct board power
   supply, and power on. USB serial is useful for boot diagnostics.
4. Wait for Linux to boot. Obtain the board's address from your network's DHCP
   leases or serial console. Replace `<BOARD_IP>` in this guide with it.
5. Confirm SSH access: `ssh xilinx@<BOARD_IP>`. Use the account/password
   configured for your image; commands below assume user `xilinx` and home
   `/home/xilinx`. Adjust both if your image uses another account.

Use the PYNQ Python environment supplied by the image. The board runtime needs
**NumPy and PYNQ**; it does not need PyTorch, snnTorch, Vivado, or the reference
training repository. FPGA access must be available to the selected interpreter.

## 3. Prepare the supplied overlay on the host

If starting on another host, clone the branch and enter it:

```bash
git clone --branch improvement_stress https://github.com/chidellapranav/spikeface.git
cd spikeface
```

For an existing checkout, start from its repository root. Replace `run_01` with
an unused name for each deployment. Keep these variables in the host shell:

```bash
export SPIKEFACE_REPO="$PWD"
export SPIKEFACE_RUN="$SPIKEFACE_REPO/bootrun/artifacts/run_01"
mkdir -p "$SPIKEFACE_RUN/deploy"
cp "$SPIKEFACE_REPO/bootrun/lfw_system.bit" "$SPIKEFACE_RUN/deploy/lfw_system.bit"
cp "$SPIKEFACE_REPO/bootrun/lfw_system.hwh" "$SPIKEFACE_RUN/deploy/lfw_system.hwh"
cp "$SPIKEFACE_REPO/bootrun/lfw_system.xsa" "$SPIKEFACE_RUN/deploy/lfw_system.xsa"
sha256sum "$SPIKEFACE_RUN/deploy/lfw_system.bit" "$SPIKEFACE_RUN/deploy/lfw_system.hwh"
```

The copies in `bootrun/` were extracted together from `lfw_system_301026.xsa`
and given the same basename, as required by our PYNQ runner. The included HWH
is the **top-level `lfw_system.hwh`**, not an interconnect's HWH.
For this supplied export, the expected SHA-256 values are:

```text
9befaddf4e87aacdb0da06ece2fdfeb2f3ae7df67c269ff1d521559efadc6ae7  lfw_system.bit
989e10cab6cf4485dfe61e68464cb186e620aedfb44b4d0ce6685a396bac28df  lfw_system.hwh
```

Regenerated hardware will have its own hashes. Keep its `.bit`/`.hwh` together.
The standalone hardware files and XSA are copied into `bootrun/`; generated
Vivado run folders are not needed for the supplied-export deployment path.

Before generating the bundle, prepare **host Python 3.10+**:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install snntorch==0.9.4 scipy
```

Obtain the reference checkout if absent:

```bash
git clone https://github.com/chidellapranav/trmp_spike.git ../trmp_spike
```

Ensure it contains all of these host inputs (a clone alone is insufficient if
model/data assets are missing):

```text
../trmp_spike/
├── 02_snn_model_train.py
├── phase1_outputs/spikeface_model.pt
└── data/spikeface_lfw/
    ├── X_test.npy
    ├── y_test.npy
    └── class_names.npy
```

Use the original checkpoint for first bring-up. The exporter loads the model
class without running training and requires the exact 53-class ordering.

## 4. Export the model bundle on the host

Run on the host from the repository root with the Python environment active:

```bash
python -m PS.export_lfw --repo "$SPIKEFACE_REPO/../trmp_spike" \
  --output "$SPIKEFACE_RUN/board_bundle"
```

This computes full scalar float32 traces for two distinct images and may take
time. It produces `board_bundle/manifest.json` and `board_bundle/lfw_bundle.npz`.
The manifest records configuration, timestep count, class ordering, checkpoint
identity, and payload hash. The NPZ contains parameters, images, labels, smoke
indices, and expected traces. The exporter refuses to overwrite a bundle.

The driver uses 224 conv1, 3,248 conv2, and 217,247 readout float32 parameters.
The NPZ readout key is `fc1`; the actual hardware block is `lfw_readout_axi_0`.
Input is 4,096 float32 pixels (16,384 bytes) and output is 53 float32 scores
(212 bytes) **per timestep**; the same grayscale frame repeats for all timesteps.

## 5. Copy the files to the running board

After SD boot, confirm SSH access. These commands use the example board user
`xilinx`; adjust it for your image. They transfer only the board runtime,
bundle, and matching overlay pair:

```bash
export SPIKEFACE_BOARD="xilinx@<BOARD_IP>"
ssh "$SPIKEFACE_BOARD" 'mkdir -p /home/xilinx/spikeface/reports'
scp -r "$SPIKEFACE_REPO/bootrun/PS" "$SPIKEFACE_RUN/board_bundle" \
  "$SPIKEFACE_BOARD:/home/xilinx/spikeface/"
scp "$SPIKEFACE_RUN/deploy/lfw_system.bit" "$SPIKEFACE_RUN/deploy/lfw_system.hwh" \
  "$SPIKEFACE_BOARD:/home/xilinx/spikeface/"
ssh "$SPIKEFACE_BOARD"
```

Expected board layout:

```text
/home/xilinx/spikeface/
├── PS/                    # lfw_driver.py, run_lfw.py, etc.
├── board_bundle/
│   ├── manifest.json
│   └── lfw_bundle.npz
├── lfw_system.bit
├── lfw_system.hwh
└── reports/
```

## 6. Program the FPGA and run golden replay

Run the following **on the board**, in its PYNQ Python environment with FPGA
access permissions. If that environment requires privilege, use its configured
privileged interpreter; a different system Python may not have PYNQ installed.

```bash
cd /home/xilinx/spikeface
python3 -c 'import numpy, pynq; print(numpy.__version__, pynq.__version__)'
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit \
  --output reports/smoke.json
```

The command programs the FPGA, allocates contiguous DDR parameters, and replays
image A, image B, then image A again. Each of the three smoke records must pass
bit-for-bit comparison of every score at every timestep. `passed` must be true.
Expected fabric clock is 50 MHz; inspect `fclk_mhz` in the JSON report.
Rebooting clears the programmed overlay; rerun the command after Linux boots.

The driver starts readout → conv2 → conv1, arms receive DMA before transmit,
and clears all recurrent state on the first timestep of each image. Do not run
another application or kernel driver that owns these DMA channels concurrently.

## 7. Evaluate five images, then the full dataset

Run on the board after smoke succeeds:

```bash
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit \
  --mode evaluate --limit 5 --output reports/five_images.json
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit \
  --mode evaluate --output reports/board_accuracy.json
```

Each command reloads the overlay and repeats smoke before evaluation. The full
report includes hardware/model hashes, predictions, targets, confusion matrix,
accuracy, and elapsed dataset time. Limited runs are marked partial. Elapsed
time includes Python orchestration. The supplied checkpoint's simulation result
was 406/554; physical-board accuracy is still to be measured. Prior test-based
checkpoint selection means this split is not an untouched final holdout.

From the **host**, collect the evidence:

```bash
scp -r "$SPIKEFACE_BOARD:/home/xilinx/spikeface/reports" "$SPIKEFACE_RUN/"
git rev-parse HEAD > "$SPIKEFACE_RUN/source_commit.txt"
git diff HEAD --binary > "$SPIKEFACE_RUN/source_changes.patch"
sha256sum "$SPIKEFACE_RUN/deploy/lfw_system.bit" \
  "$SPIKEFACE_RUN/deploy/lfw_system.hwh" "$SPIKEFACE_RUN/deploy/lfw_system.xsa" \
  "$SPIKEFACE_RUN/board_bundle/lfw_bundle.npz" > "$SPIKEFACE_RUN/artifact_sha256.txt"
```

Record source identity before building as well if edits may occur during the
run. Retain logs, HLS reports, routed reports, bundle, hashes, and board JSON
together. Local generated outputs under `bootrun/artifacts/` are Git-ignored.

## 8. Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing IP or board part | Check `component.xml` in each override directory and `board.xml` in the board repository; see the PL guide |
| Missing `.bit` despite an XSA | Inspect `vivado.log`, timing and DRC; the early XSA is not proof of completed implementation |
| PYNQ import/permission error | Select the board's PYNQ interpreter and FPGA-access environment |
| Missing overlay IP attribute | Check HWH instance names: `lfw_conv1_axi_0`, `lfw_conv2_axi_0`, `lfw_readout_axi_0`, `axi_dma_0`, `axi_dma_1` |
| DMA error, timeout, or short transfer | Stop, check DDR/AXI connections, DMA direction/width, score TLAST, clock and matching HWH; reload by rerunning the command |
| Need a longer diagnostic wait | Add `--timeout 120` (seconds per wait); investigate the cause instead of treating timeout growth as a fix |
| Golden mismatch | Check source/IP identity, parameter packing, checkpoint/configuration, state reset and float operator settings; retain bitwise comparison |
| Wrong predictions after smoke passes | Check dataset preprocessing, class ordering and checkpoint; measure actual accuracy |

After any interface change, recheck exported HLS register headers/HWH against
`PS/lfw_driver.py`: parameter pointer low/high `0x10/0x14`, beta `0x1c`, conv
threshold `0x24`, conv reset `0x2c`, and readout reset `0x24`.

## 9. Recommended: regenerate hardware after cloning

Use Vivado/Vitis HLS **2022.2** with Zynq-7000 device support for
`xc7z020clg400-1`. The target is PYNQ-Z2 at 50 MHz.

### A. Rebuild the saved project in Vivado

Follow [PL_Design/README.md](../PL_Design/README.md) to unpack the three
`hls_ip/*/export.zip` archives, install/register the PYNQ-Z2 board preset, and
open `PL_Design/lfw_pynq_z2_build/lfw_pynq_z2.xpr`.

In **Settings → IP → Repository**, register directories containing
`component.xml`:

| IP | Repository path relative to the clone |
| --- | --- |
| `lfw_conv1_axi` | `hls_ip/lfw_conv1/ip` |
| `lfw_conv2_axi` | `hls_ip/lfw_conv2/ip` |
| `lfw_readout_axi` | `hls_ip/lfw_readout/ip` |

Validate the block design, generate output products, run synthesis and
implementation, review timing/DRC, and generate the bitstream. Use **File →
Export → Export Hardware**, enable **Include bitstream**, and save a new XSA.
From the Tcl Console, after successful bitstream generation, an equivalent
export is (replace the output path):

```tcl
open_run impl_1
write_hw_platform -fixed -include_bit -force -file /absolute/path/new_lfw_system.xsa
```

The generated standalone bitstream is
`PL_Design/lfw_pynq_z2_build/lfw_pynq_z2.runs/impl_1/lfw_system_wrapper.bit`;
the top-level HWH is
`PL_Design/lfw_pynq_z2_build/lfw_pynq_z2.gen/sources_1/bd/lfw_system/hw_handoff/lfw_system.hwh`.
Copy them to `$SPIKEFACE_RUN/deploy/lfw_system.bit` and
`$SPIKEFACE_RUN/deploy/lfw_system.hwh`, and retain the new XSA alongside them.
Regenerate the pair together, then refresh the hardware copies in `bootrun/`
when you intend to distribute the new build.

### Alternative: rebuild IPs and PL from the optimized sources

For a fresh scripted build, register the PYNQ-Z2 preset so
`PL_Design/board_files/pynq-z2/A.0/board.xml` exists. Load your tool environment
(adapt these example installation paths):

```bash
source /home/pranavsai/WORK/Vivado/2022.2/settings64.sh
source /home/pranavsai/WORK/Vitis_HLS/2022.2/settings64.sh
command -v vivado
command -v vitis_hls
```

### B. Build and verify the optimized HLS IPs

Use the optimized sources for all three IPs in this run. This separate fresh
build does not reset the existing GUI projects. The bundled fixture is small;
its passing result does not establish full-dataset accuracy or complete reset
and backpressure coverage.

```bash
export SPIKEFACE_HLS_BUILD_DIR="$SPIKEFACE_RUN/hls"
export SPIKEFACE_HLS_STAGE=all
export SPIKEFACE_COSIM_TEST=stress
export SPIKEFACE_COSIM_STALL=0
export SPIKEFACE_CLOCK_NS=20
export SPIKEFACE_FP_ADD_IMPL=fabric
unset SPIKEFACE_HLS_BLOCK
vitis_hls -f "$SPIKEFACE_REPO/newarch_HLS/build_lfw_hls.tcl" \
  > "$SPIKEFACE_RUN/hls.log" 2>&1
```

`all` runs C simulation, synthesis, RTL cosimulation, and IP export for each
top. Inspect the log and each solution's reports under `sol/sim/report/` and
`sol/syn/report/`; require all three blocks to complete successfully before PL
integration. RTL cosimulation can take substantial time. Further stall/chain
campaign commands and evidence limits are in
[newarch_HLS/README.md](../newarch_HLS/README.md).

The IP repository paths for **this run** are:

| IP | Directory containing `component.xml` |
| --- | --- |
| `lfw_conv1_axi` | `$SPIKEFACE_RUN/hls/lfw_conv1_axi/sol/impl/ip` |
| `lfw_conv2_axi` | `$SPIKEFACE_RUN/hls/lfw_conv2_axi/sol/impl/ip` |
| `lfw_readout_axi` | `$SPIKEFACE_RUN/hls/lfw_readout_axi/sol/impl/ip` |

### C. Generate the bitstream, HWH, and XSA

Set the three IP overrides explicitly so the integration script uses the IPs
just built. `bit` creates a new project and runs the complete PL flow. Its output
directory must be fresh and outside `PL_Design/`.

```bash
export SPIKEFACE_BOARD_REPO="$SPIKEFACE_REPO/PL_Design/board_files"
export SPIKEFACE_CONV1_IP="$SPIKEFACE_RUN/hls/lfw_conv1_axi/sol/impl/ip"
export SPIKEFACE_CONV2_IP="$SPIKEFACE_RUN/hls/lfw_conv2_axi/sol/impl/ip"
export SPIKEFACE_READOUT_IP="$SPIKEFACE_RUN/hls/lfw_readout_axi/sol/impl/ip"
vivado -mode batch -source "$SPIKEFACE_REPO/PL_Design/create_lfw_pl.tcl" \
  -tclargs bit "$SPIKEFACE_RUN/pl" > "$SPIKEFACE_RUN/vivado.log" 2>&1
```

Require synthesis/implementation success, nonnegative routed setup and hold
slack, and no error-severity DRC violations. Review unconstrained paths and
warnings as well. The script checks timing/DRC before writing the bitstream.
An XSA can exist from the earlier block-design stage even if implementation
later fails; require the completed `.bit`/`.hwh` exports and success log.

| Generated artifact | Host path |
| --- | --- |
| Vivado project | `$SPIKEFACE_RUN/pl/lfw_pynq_z2.xpr` |
| FPGA overlay | `$SPIKEFACE_RUN/pl/lfw_system.bit` |
| Matching hardware metadata | `$SPIKEFACE_RUN/pl/lfw_system.hwh` |
| Hardware platform including bitstream | `$SPIKEFACE_RUN/pl/lfw_system.xsa` |
| Routed timing report | `$SPIKEFACE_RUN/pl/timing_summary.rpt` |
| Resource report | `$SPIKEFACE_RUN/pl/utilization.rpt` |
| Design-rule report | `$SPIKEFACE_RUN/pl/drc.rpt` |

Keep the `.bit` and `.hwh` from the **same build** with the same basename.
PYNQ loads that pair; the XSA is retained for hardware handoff and traceability.
The previous 40-class XSA under `Spike_old/` uses a different interface.

After the scripted build succeeds, select its exports for sections 5–7:

```bash
cp "$SPIKEFACE_RUN/pl/lfw_system.bit" "$SPIKEFACE_RUN/deploy/lfw_system.bit"
cp "$SPIKEFACE_RUN/pl/lfw_system.hwh" "$SPIKEFACE_RUN/deploy/lfw_system.hwh"
cp "$SPIKEFACE_RUN/pl/lfw_system.xsa" "$SPIKEFACE_RUN/deploy/lfw_system.xsa"
```

Run `python -m unittest discover -s tests -v` on the host to check the software.
Keep build logs, timing/resource/DRC reports, hardware hashes, model bundle,
and board results together. This documentation update inspected the supplied
archive and bitstream; it did not rerun implementation or execute on a board.

Related guides: [PL setup](../PL_Design/README.md),
[PS board runner](../PS/LFW_BOARD_RUN.md),
[optimized HLS verification](../newarch_HLS/README.md).
