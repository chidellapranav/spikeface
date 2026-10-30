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
counterexamples to obtain a PASS. Legacy tools and evidence: [../Spike_old/stress_sim/README.md](../Spike_old/stress_sim/README.md).
