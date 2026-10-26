# SpikeFace LFW verification and accuracy report

**Date:** 2026-10-24\
**Authoritative Python source:** `trmp_spike`, commit `7a4d1f4`\
**Original HLS baseline:** `spikeface`, commit `a92dc8025db1fab4d6e494804a876e58e4dde3cd`\
**Release decision:** **not ready for final PL integration**. Architecture and
dataset classification agreement are established in C++ simulation; strict
PyTorch numerical equivalence, completed RTL verification, board validation,
routed WNS and the >96% accuracy target remain unfulfilled.

## 1. Results that matter

| Check | Result | Interpretation |
|---|---|---|
| Original supplied PyTorch checkpoint, complete test dataset | **406/554 = 73.2852%** | Reproduces the repository's recorded baseline |
| New LFW HLS C++ / connected AXI chain, complete test dataset | **406/554 = 73.2852%** | Preserves measured accuracy of this checkpoint |
| HLS versus PyTorch class predictions | **554/554 agree** | Classification agreement on this dataset, not universal equivalence |
| Strict final score tolerance, atol=rtol=1e-4 | **157 images fail** | Numerical equivalence gate fails |
| Maximum final score absolute difference | **0.051670074462890625** | Small arithmetic differences can change recurrent spike trajectories |
| Independent scalar float32 specification | **Bit-exact in tested cases** | HLS agrees with the explicitly defined sequential reduction contract |
| LFW stress / counterexample campaign | **3,946 checks; 12,561,374 values; 95 failed checks** | All failures are retained PyTorch trace comparisons; scalar checks pass |
| LFW mutation tests | **4/4 deliberate defects detected** | Reset, strict threshold, temporal mean and TLAST checks are sensitive |
| Legacy fixed-point campaign | **8,651 checks; 18,254,657 values; zero failures** | Applies to the separate 40-class legacy architecture |
| Original legacy C++ testbenches | **12/12 pass** | Includes two duplicated testbench pairs |
| Application / PS tests | **26 pass** | Includes HTTP, labeled image datasets, validation, DMA mocks and both PS drivers |
| Native Vitis HLS C simulation / synthesis | **All three LFW tops pass** | Generates RTL; does not establish routed timing |
| Native RTL co-simulation | **Incomplete, explicitly stopped** | No completed RTL transaction result; repeated DSP OPMODE warnings unresolved |
| Readout refit candidate, PyTorch | **427/554 = 77.0758%** | Validation-selected improvement, still below target |

The >96% requirement means at least **532 correct out of 554**. Neither model
meets it. Raising the clock, changing WNS, or correcting the PS scheduler cannot
convert a 73% trained classifier into a 96% classifier while preserving its
predictions.

## 2. Why the original hardware was not the supplied Python model

| Property | Legacy HLS | Authoritative LFW model / new HLS |
|---|---|---|
| Input | 32×32 binary Bernoulli spike frame | 64×64 normalized grayscale, constant every timestep |
| Classes | 40 | 53 |
| Convolution bias | Absent | Present |
| Batch normalization | Absent | Present after convolution / linear |
| Pooling | Average pooling after LIF | Max pooling before LIF |
| Hidden state | Q3.5 saturation and ties-to-even | Float32, no membrane clipping |
| Firing condition | `mem >= threshold`, usually 1.0 | `mem > threshold`, hidden threshold 0.5 |
| Hidden reset | Immediate reset to zero | Delayed subtractive reset based on previous membrane |
| Classifier | 1024→64 spiking→40 accumulating | 4096→53 linear + BN + leaky readout |
| Temporal result | Accumulated class votes | Mean of the leaky readout membrane trace |
| Default timesteps | 50 deployment / 20 golden | 25 |

The original checkpoint cannot be loaded into the legacy HLS. The six old IPs
and their XSA are historical artifacts, not a deployable version of this LFW
model. New `lfw_model.cpp`, `lfw_axi.cpp` and `lfw_model.h` implement the agreed
architecture. The source Python training script and its original weights were
left unchanged.

## 3. What was built and how it was checked

The new design has three tops: `lfw_conv1_axi`, `lfw_conv2_axi`, and
`lfw_readout_axi`. It uses float32 so it does not silently impose the legacy
8-bit quantization on a model trained in floating point. Parameters reside in
DDR through AXI masters: the 53×4096 readout weights alone require 868,352 bytes,
which exceeds the Zynq-7020's on-chip BRAM capacity. BN is exported as scale and
shift and applied after the original convolution result, before max pooling.

The C++ harness uses the actual Xilinx Vitis HLS 2022.2 types and stream
implementation. It runs both plain-array and connected AXI paths. Python uses
an independent, explicit float32 reduction order; multiply-add contraction is
disabled in the C++ compilation. Every spike, membrane, readout evidence and
score is checked. Scalar score and membrane comparisons include all float bits,
not only decimal tolerance. Packets require TLAST at output 53 and TKEEP/TSTRB=15.

Stress coverage includes:

- Original checkpoint images 0–4 and 267, over all 25 timesteps. Images 4 and
  267 are deliberate retained counterexamples from the full-dataset run.
- All three stages individually and the full connected network.
- Zero, dense, random and repeated inputs; independent-layer and whole-image resets.
- Eight seeded synthetic parameter sets, including signed/zero BN scales and
  beta values 0, .9375 and 1.
- Exactly-at-threshold inputs, to distinguish strict `>` from `>=`.
- Delayed subtractive reset versus immediate reset, and temporal mean versus sum.
- Four isolated mutations, all detected without altering the working sources.

The complete dataset run executes actual C++ plain and AXI blocks on every
image, compares their outputs and packets, and compares classification and
scores with the independently run original PyTorch model. C++ simulation time
is not an FPGA latency benchmark.

## 4. Numerical divergence: concrete counterexample

For test image **267**, the first eight timesteps have matching spikes, with
conv1 membrane differences growing from roughly 1.55e-6 to 1.03e-5. At timestep
**8** one conv1 spike differs, and three conv2 spikes differ. At timestep 9 the
different reset history produces a conv1 membrane difference near 0.5. The final
score error reaches **0.051670074462890625**. The final class still agrees.

This is consistent with backend-dependent float32 convolution/reduction and BN
rounding differences amplified by discontinuous spike thresholds and recurrent
state. It is not corrected by changing a comparison tolerance after observing
failure. The exact operation responsible for every first differing bit has not
been established. The scalar Python contract and C++ contract agree, while the
original PyTorch backend does not always agree with those traces.

**Bit-reference verdict:** the legacy HLS is bit-referenced to its committed
golden fixtures, not to the supplied LFW architecture. The new LFW HLS is
bit-referenced to the independent scalar float32 specification in tested cases,
but is **not bit-exact to the supplied PyTorch execution**. Matching all 554
predicted classes does not change that verdict.

Next, either establish a canonical numerical/quantized reference and measure
its accuracy before hardware freeze, or reproduce a specified PyTorch
operator/reduction contract exactly. Quantization-aware training is a reasonable
candidate path, but no unmeasured quantized accuracy is claimed here. This is a
real release decision, not a reason to label the HLS flawless.

## 5. Accuracy improvement experiment

The supplied best checkpoint reproduces **406/554**. Its training loop chose
the best epoch using the test split, so that split has already influenced model
selection. The claimed historical 93.75% ORL result belongs to a different,
unrecovered training workflow and cannot support LFW accuracy.

A separate readout refit froze all hidden layers and retained the same model
architecture. Temporal spike features were computed from training images.
Seven ridge regularization strengths were compared on a stratified validation
split formed only from the 2215 training images. Alpha=1000 was selected with
validation accuracy **71.1061%**. The readout was then refitted using all training
images, saved separately, and evaluated once on the supplied test split:
**427/554 = 77.0758%**, an increase of **21 correct images / 3.7906 percentage
points** over the original.

`models/lfw_readout_refit.pt` and `models/readout_refit_report.json` preserve the
candidate and its provenance. This candidate has been evaluated in PyTorch;
its entire-dataset HLS comparison has not been run. It is not silently substituted
for the checkpoint in the hardware-agreement study. Its inherited hidden layers
still come from a test-selected checkpoint, so this is a development experiment,
not an independent final accuracy claim.

Further work toward >96% requires an honest training/validation/final-test
protocol and likely a stronger data/training strategy. The original sparsity
penalty uses `.item()` statistics, which do not propagate a sparsity gradient.
That training issue should be fixed in controlled experiments. It does not
explain away the measured accuracy gap or justify reporting the target as met.

## 6. Native HLS and timing

Native Vitis HLS C-simulation and synthesis succeeded for all three new tops.
At the selected 20 ns / 50 MHz target:

| Top | HLS estimated period | BRAM18K | DSP | LUT |
|---|---:|---:|---:|---:|
| conv1 | 14.600 ns | 25 | 7 | 6208 |
| conv2 | 14.600 ns | 14 | 7 | 6280 |
| readout | 14.600 ns | 5 | 5 | 4566 |

These are per-IP synthesis estimates. They exclude PS/DMA/interconnect overhead
and are not a placed-and-routed fit/timing result. Loop II requests are not all
achieved because of float accumulation dependencies and memory access; reduced
throughput is documented, consistent with the user's latency preference.

The initial 10 ns attempt estimated conv1 at 10.053 ns, so it was not advertised
as closing at 100 MHz. The conservative 50 MHz target is used instead. Physical
acceptance remains **routed WNS ≥ 0 ns**. No routed WNS has been measured.

Generated-Verilog co-simulation was attempted using a two-timestep fixture.
The run was explicitly stopped while still simulating conv1 because it was
very slow and emitted repeated DSP primitive OPMODE warnings. It did not produce
a completed RTL transaction result. `lfw_cosim_results.json` records the state;
the raw ignored log remains available locally. These warnings have not been
proved harmless, and no RTL or board pass is claimed.

## 7. PS and PL scope

The CLI now accepts the bundled NumPy dataset or a labeled database of aligned
64×64 images, and produces measured accuracy, per-class results, confusion
matrix and model/data hashes. `--backend software` runs the original model;
`--backend hls` runs the new C++/AXI simulation; `--backend pynq` is wired to the
candidate new-board driver and requires matching future hardware.

`PS/lfw_driver.py` outlines and implements parameter DDR buffers, float register
codes, downstream-first scheduling, first-timestep resets, checked DMA lengths,
cache completion and timeouts. Its mocks pass; board behavior is unverified.
The legacy driver was separately corrected for pre-cast input validation,
missing cache completion, image isolation and DMA errors.

The new PL topology is specified in [PS_PL_PLAN.md](../docs/PS_PL_PLAN.md).
No new integrated bitstream is represented as ready. Numerical-contract
acceptance, completed RTL checks, routed nonnegative WNS and golden replay on
the board must precede a hardware accuracy claim.

## 8. Reproducibility and evidence

Use the commands in the root README and `stress_sim/README.md`. The include path
used locally is the real Vitis HLS 2022.2 installation; Python uses CPU PyTorch
2.5.1 and snnTorch 0.9.4. Seeds and source/checkpoint hashes are recorded.

| Artifact | Evidence |
|---|---|
| `lfw_evaluation.json` | Original PyTorch baseline, predictions and confusion matrix |
| `lfw_hls_evaluation.json` | Complete dataset C++/AXI agreement and failed score tolerance |
| `lfw_results.json` | Every stress comparison, including 95 retained failing checks |
| `lfw_divergence.json` | Failed image indices and timestep diagnostics |
| `lfw_mutation_results.json` | Four deliberately incorrect versions rejected |
| `lfw_synthesis_50mhz_results.json` | Native synthesis estimates |
| `lfw_cosim_results.json` | Incomplete RTL run and unresolved warning evidence |
| `lfw_refit_evaluation.json` | Candidate's 427/554 PyTorch result |
| `../models/readout_refit_report.json` | Training-only validation selection and candidate provenance |
| `REPORT.md` | Separate legacy fixed-point verification and findings |

The strict verification commands return nonzero while numerical equivalence
fails. The classification and scalar-reference successes must not be used to
override that gate without an explicit, documented numerical acceptance decision.
