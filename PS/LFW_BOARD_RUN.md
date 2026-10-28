# Run the LFW prototype on PYNQ-Z2

The PL must implement the three LFW IPs in `create_lfw_pl.tcl`, with the
register layout in `lfw_driver.py`. Use its matching bitstream and HWH.

## 1. Export on the host

Use the host environment that already runs the reference checkpoint (NumPy,
Pillow, PyTorch and snnTorch). From the repository root:

```bash
python -m PS.export_lfw --repo ../trmp_spike --output board_bundle
```

Use `--checkpoint PATH` to explicitly select another compatible checkpoint.
Export computes full 25-timestep scalar-reference traces for two distinct
images; this can take time. The bundle contains packed float32 weights,
biases and BN affine parameters, the test images and labels, reference traces,
configuration, checkpoint identity and a payload hash. Choose a fresh output
directory for each export.

## 2. Copy to the board

Copy the `PS` directory and `board_bundle` directory into, for example,
`/home/xilinx/spikeface/`. Put the matching hardware files alongside them as
`lfw_system.bit` and `lfw_system.hwh`. Extract the HWH from the matching XSA if
it was not exported separately. Both files must have the same basename.

The board runner only imports NumPy, PYNQ and Python standard-library modules.
It does not need PyTorch, snnTorch, the training repository or internet access.
Run inside the board's PYNQ Python environment, with its usual FPGA permissions.

## 3. Golden replay first

From `/home/xilinx/spikeface`:

```bash
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --output smoke.json
```

This programs the FPGA, loads contiguous DDR parameters, and executes A, B, A
with fresh state per image. Every float32 score at every timestep must match
the scalar contract bit for bit. This compares against the hardware numerical
contract; PyTorch scores can differ because of reduction order. Any mismatch
produces a failed report and a nonzero exit. Do not loosen the comparison to
hide a mismatch; inspect the interface, parameter packing and numerical build.

## 4. Dataset evaluation

```bash
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --mode evaluate --limit 5 --output five_images.json
python3 -m PS.run_lfw --bundle board_bundle --bitstream lfw_system.bit --mode evaluate --output board_accuracy.json
```

Each evaluation repeats golden replay first. Reports include predictions,
scores, targets, confusion matrix, measured accuracy, elapsed dataset time,
fabric clock and hardware/model hashes. Limited runs are marked partial.
Elapsed time includes Python orchestration and is not accelerator-only latency.
The original checkpoint measured 406/554 in simulation; board accuracy must be
measured. These test images were previously used for checkpoint selection.

On DMA error or timeout, stop and rerun the command to reload the overlay.
`--timeout SECONDS` changes the per-wait timeout (default 60 seconds). Check IP
instance names, offsets against exported headers/HWH, fabric clock, DMA widths,
DDR connectivity and score TLAST if bring-up stalls. Never run another program
that owns these DMA channels at the same time.

Host-side checks (with NumPy installed):

```bash
python -m unittest discover -s tests -v
```
