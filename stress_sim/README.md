# SpikeFace HLS verification

**Current authority:** the owner selected the 64×64 / 53-class LFW model.
Read [LFW_REPORT.md](LFW_REPORT.md) for the current release decision. The original
fixed-point campaign described below remains a separate legacy verification.
The native RTL continuation and its current evidence are recorded in
[RTL_RESUME_REPORT.md](RTL_RESUME_REPORT.md).

The new campaign is:

```bash
python3 stress_sim/lfw_verify.py --repo ../trmp_spike
python3 stress_sim/lfw_mutations.py --repo ../trmp_spike
python3 stress_sim/lfw_dataset.py --repo ../trmp_spike
```

These commands require CPU PyTorch and snnTorch as well as NumPy and Xilinx
headers. In this workspace, prefix them with
`PYTHONPATH=stress_sim/build/python_deps` to use the installed local dependencies.
Run `python3 -m app lfw-evaluate --repo ../trmp_spike --output
stress_sim/lfw_evaluation.json` first to reproduce the baseline used by the full
HLS dataset comparison. The strict verification and dataset checks currently
exit **1** for genuine PyTorch numerical divergence. The independent scalar
float32 and classification results are reported separately; do not remove the
counterexamples to obtain a PASS. The legacy commands below still pass.

This folder compares the actual HLS C++ against an independent NumPy integer
specification, using the real Vitis HLS fixed-point and stream headers. It does
not replace Xilinx types with simplified C++ arithmetic.

## Run

From the repository root, with Python 3, NumPy, and g++ installed:

```bash
export XILINX_HLS_INCLUDE=/path/to/Vitis_HLS/2022.2/include
python3 stress_sim/run.py --cases 32 --seed 20261024
python3 stress_sim/original_tests.py
python3 stress_sim/mutations.py
```

The include path defaults to the installation discovered on this machine.
`run.py` compiles the bridge before testing; `--no-build` is only for an already
compiled, current harness. A mismatch or execution error returns a nonzero exit
status. `--output` selects a JSON report filename. Keep seed and source hashes
with any published result. More random cases can be requested with `--cases`.

## Artifacts

| File | Purpose |
|---|---|
| `reference.py` | Independent, exact integer numerical specification and golden parser |
| `harness.cpp` | Binary bridge calling real plain-array and AXI HLS implementations |
| `run.py` | Golden, exhaustive boundary, randomized, adversarial and reset tests |
| `original_tests.py` | Compile and run all 12 original testbench files |
| `mutations.py` | Introduce four deliberate defects in isolated copies and require detection |
| `baseline.json` | Pre-fix run: arithmetic passed; uninitialized TSTRB was detected |
| `results.json` | Post-fix differential results and source hashes |
| `original_results.json` | Original testbench outcomes |
| `mutation_results.json` | Evidence that tests reject deliberately incorrect implementations |
| `REPORT.md` | Human-readable results, findings, limits and next steps |

Build products, logs, and isolated mutation copies are ignored by Git. Structured
JSON reports are retained. The harness sends raw integer fixed-point codes, not
floating-point formatted numbers. The connected-chain test runs two paths:
plain-array stages feed their next plain-array stage, and actual AXI stage
outputs feed the next AXI stage. Every boundary is compared with Python.

## Numerical contract

Currents use signed 20-bit Q8.12 codes. Every accumulator assignment saturates;
the result can differ from a floating-point or unbounded integer dot product.
Membrane assignment rounds ties to even and saturates to signed Q3.5, followed
by threshold comparison and reset-to-zero. Pooling codes are sums of four
binary inputs, representing quarters. fc2 uses Q2.6 weights and emits Q8.24 raw
score bits. Its output widens precision, not range: saturation can still lose
ranking information. The model preserves this behavior rather than silently
changing the trained algorithm.

The deployment model starts each image from zero state. HLS `clear_state` must
be asserted for conv1/conv2/fc1 together with fc2 `clear_accum` on its first
timestep. Default C++ arguments preserve compatibility with original testbenches;
the exported hardware interface has changed and must be rebuilt.

## Scope

These are C-level functional tests, including the HLS AXI packet structure.
They do not simulate RTL backpressure, finite FIFO deadlocks, DDR, AXI-Lite bus
transactions, DMA caches, physical timing, or the board. `build_hls_ips.tcl`
provides the next HLS tool stages. No finite stress suite proves an HLS design
flawless. Missing original Python training/quantization code prevents a universal
claim of equivalence to that implementation. Missing held-out data prevents an
accuracy claim. See `REPORT.md` before proceeding to PL.
