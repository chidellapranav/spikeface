# LFW readout candidate

`lfw_readout_refit.pt` preserves the supplied 53-class architecture and all hidden
convolutional weights. Only the linear readout and its batch-normalization
parameters were refitted. The original checkpoint remains untouched in the
reference repository.

Regularization was selected on a stratified training-only validation split,
then the selected readout was fitted using all 2215 training images. See
`readout_refit_report.json` for indices, candidate scores, seed and source hash.

| Model | Correct / test images | Accuracy |
|---|---|---|
| Original supplied checkpoint | 406 / 554 | 73.2852% |
| Readout candidate | 427 / 554 | 77.0758% |

The candidate was evaluated once after validation selection; its test scores
were not used to choose regularization. The frozen original checkpoint had
already been selected using this same test split, so these are development
results, not a new untouched-holdout study. **The >96% target is not achieved.**

Run from the repository root:

```bash
python3 -m app lfw-evaluate --repo ../trmp_spike \
  --checkpoint models/lfw_readout_refit.pt --output candidate_accuracy.json
```

Requires CPU PyTorch and snnTorch; see the main README. This is a software
checkpoint; it is not a bitstream and does not make the legacy XSA compatible.
