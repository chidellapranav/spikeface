# Deploying to the PYNQ-Z2 and measuring accuracy

The PL design and the PS driver are both written. This document covers getting them
onto the board, the problems that will corrupt an accuracy measurement if left
alone, and how to get the numbers.

**Nothing here has been run on hardware yet.** Claims are labelled as read from a
file, derived by arithmetic, or unverified.

---

## 1. Boot and load flow (SD card)

The SD card holds only the generic PYNQ Linux image. Your design is loaded from
Python after boot. No custom boot image is needed.

1. **Flash the SD card.** Download the PYNQ-Z2 image from pynq.io/boards. Write it to
   a 16 GB or larger microSD card with Balena Etcher.
2. **Boot.** Set the board's boot jumper to SD, insert the card, connect Ethernet,
   power on. Jupyter is at `http://pynq:9090` (or `http://<board-ip>:9090`),
   password `xilinx`.
3. **Get the PL files.** `spikeface/system_150926.xsa` in this repo already contains
   the bitstream and hardware handoff:
   ```bash
   unzip spikeface/system_150926.xsa system_150926.bit system_top.hwh
   mv system_150926.bit spikeface.bit
   mv system_top.hwh    spikeface.hwh      # both files must share a base name
   ```
   Do this after fixing problem 3.1 below and re-exporting, not before.
4. **Copy to the board**, into `/home/xilinx/spikeface/`, with `scp` or the Jupyter
   upload button: `spikeface.bit`, `spikeface.hwh`, `spikeface_driver.py`, the four
   quantized weight files, and the test data.
5. **Run.** In a Jupyter notebook, or `sudo python3 script.py` over SSH (loading an
   overlay outside Jupyter needs root).
6. **First checks.**
   ```python
   from pynq import Overlay, Clocks
   ol = Overlay("/home/xilinx/spikeface/spikeface.bit")
   print(list(ol.ip_dict))      # expect the six *_axi_0 IPs and axi_dma_0/1
   print(Clocks.fclk0_mhz)      # see problem 3.2
   ```

---

## 2. Problem 1: membrane state leaks from one image into the next

**What happens.** conv1, conv2 and fc1 keep membrane potentials in `static` arrays
with no reset input. Only fc2 has one (`clear_accum`). Every golden-vector test
started from zeroed state, so two images back to back were never tested. On the
board, image N+1 starts with image N's leftover potentials, which skews any
multi-image accuracy figure.

**Why running empty timesteps does not help.** With `beta` at code 30/32 and
round-half-to-even, the leak update `m -> round(30m/32)` maps any code up to 8 back
to itself (for example 8 -> 7.5 -> 8). Residual potential up to 0.25 therefore never
decays to zero. This is derived from the rounding arithmetic, not measured.

**Fix A: available now, no HLS change.** Reprogram the PL before each image.
BRAM initialises to zero on configuration. The driver does this with
`run_inference(..., fresh_state=True)` or `reset_pl_state()`, which reload the
overlay and rewrite all weights and parameters. It costs on the order of seconds per
image; for 80 images that is minutes.

**Fix B: the proper fix, not implemented.** Add a `clear_state` input to conv1, conv2
and fc1, the same pattern as fc2's `clear_accum`: on the first timestep, read the
membrane as 0 instead of from `mem_state`. It needs HLS edits, re-synthesis,
re-packaging and Vivado re-integration for three blocks. Do this before final results.

**Check that Fix A works (do this once).** Use fixed spike frames so encoding noise
is out of the picture:
```python
r1 = sf.run_inference(spike_frames=F, T=20, return_trace=True)                   # first run after boot
r2 = sf.run_inference(spike_frames=F, T=20, return_trace=True)                   # no reset
r3 = sf.run_inference(spike_frames=F, T=20, return_trace=True, fresh_state=True)
# r2 differing from r1 shows the leak is real.
# r3 equal to r1 shows the reset works.
```

---

## 3. Problems found in the committed hardware (fix before measuring)

Read from `spikeface/rebuild_vivado.tcl` and the `system_top.hwh` inside the XSA.

### 3.1 fc1's weights cannot be reached from the PS

- fc1's weight array is on its AXI-Lite interface at IP offset `0x10000`-`0x1FFFF`
  (`XFC1_LAYER_AXI_CTRL_ADDR_WEIGHTS_BASE = 0x10000`, depth 65,536).
- The block design gives fc1's `s_axi_CTRL` only `0x40020000`-`0x4002FFFF` (64 KiB).
- The weights therefore sit at `0x40030000`-`0x4003FFFF` absolute, which is the
  window assigned to **fc2**. They are outside fc1's window and overlap fc2's.
- Result if unfixed: fc1 keeps all-zero weights, never spikes, fc2 sees only zeros,
  every score is 0 and `argmax` is always class 0, about 2.5% accuracy (2 of 80).

The driver's `configure()` checks the mapped window and refuses to continue with an
explanatory error instead of failing silently.

**Fix (Vivado only, no HLS change):** Address Editor -> `fc1_layer_axi_0/s_axi_CTRL/Reg`
-> set Range to 128 K and Offset to a 128 K-aligned address that does not overlap
anything (for example `0x40040000`) -> Validate Design -> regenerate the bitstream ->
re-export the XSA / `.bit` / `.hwh` and commit them.

Note on the interface: the repo's `fc1_layer_axi.cpp` and this bitstream both use the
AXI-Lite weight interface. A BRAM-interface variant was discussed earlier but is not
in the build, and would additionally need a memory IP wired to the port. Keeping
AXI-Lite and fixing the range is the simpler path.

### 3.2 The fabric clock is 50 MHz, not 100 MHz

`system_top.hwh` records `PCW_FPGA0_PERIPHERAL_FREQMHZ = 50`, and the reset block is
named `rst_ps7_0_50M`. The HLS blocks were synthesized for a 10 ns clock.

- Correctness is unaffected. Accuracy does not depend on the clock.
- **Every latency figure roughly doubles** relative to the 100 MHz estimates.
- The reported post-route `WNS = +7.010 ns` was measured against a 20 ns period. It
  implies a worst path of about 13 ns, so it does **not** show the design closes at
  100 MHz. Moving to 100 MHz means re-running implementation and checking again.

Report latency at the clock the board actually runs (`Clocks.fclk0_mhz`).

---

## 4. Measuring accuracy

**Software baseline.** 93.75% (75 of 80 test images) at T=50 in the Python study,
for both the float-weight SNN and the 8-bit fixed-point SNN. The test set is the ORL
8-train / 2-test-per-identity split (80 images).

Prerequisites: problems 3.1 fixed, Fix A available, the quantized weights and test
data on the board.

### Step A: golden replay (functional check, do first)
Export the spike frames and the expected fc2 score trace from the Python
fixed-point reference for one image (the golden-vector export already computes both).
On the board, right after loading the overlay (state is zero):
```python
pred, scores, ex = sf.run_inference(spike_frames=F, T=20, return_trace=True)
np.testing.assert_allclose(ex["trace"], golden_scores, atol=2**-24)
```
Matching scores at every timestep means the PL, DMA, driver and decoding are all
correct. Any accuracy number is meaningless until this passes.

### Step B: test-set accuracy with pre-generated spikes (like-for-like)
Generate the T=50 spike trains for all 80 test images on the PC with a seeded encoder
and save them. On the board, run each with `fresh_state=True`, take the `argmax`, and
compare with the label. Accuracy is correct / 80. Input spikes are identical to the
software run, so any difference comes from the hardware or quantization. Also compare
scores image by image against the Python fixed-point scores, which is more sensitive
than the accuracy percentage.

### Step C: on-board encoding (deployment realism)
Repeat with the driver's own random encoder (`run_inference(image01, ...)`). Results
vary run to run; report the mean and standard deviation over 5-10 runs.

**Reading the result.** One image is 1.25 percentage points, and 75/80 has a wide
confidence interval (roughly +/-5 points). A difference of one or two images is within
noise. Use Step B's paired comparison to judge equivalence.

---

## 5. Other numbers to record in the same loop

| Metric | How |
|---|---|
| Latency per image | `time.perf_counter()` around `run_inference`. From the synthesis reports the six blocks total about 1.13 M cycles per timestep (conv2 is about 1.03 M of it): about 22.6 ms per timestep at 50 MHz, about 1.1 s per image at T=50 (half that at 100 MHz). This is an estimate; use the measured value. |
| Throughput | images per second, from the above |
| Sparsity | `run_inference(..., collect_taps=True)` returns per-layer sparsity from the `total_active_taps` registers |
| Energy per inference | power x latency. The 1.766 W from Vivado is an estimate; an inline USB power meter gives a measured figure. |

---

## 6. Order of work

1. Fix 3.1 in Vivado, regenerate, re-export `.bit`/`.hwh`.
2. Decide whether to keep 50 MHz or move to 100 MHz (3.2).
3. Flash the SD card, boot, copy files, confirm `ip_dict` and `Clocks.fclk0_mhz`.
4. Step A (golden replay). Fix whatever it exposes.
5. Run the Fix A check (section 2).
6. Step B, then Step C, then the metrics in section 5.
7. Before final results, implement Fix B (`clear_state`) so results do not depend on reprogramming the PL.
