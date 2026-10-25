# LFW RTL verification continuation, 2026-10-25

The native Vitis HLS 2022.2 campaign uses the production LFW tops, a 20 ns
clock constraint, fabric floating-point add/sub operators, and the exported
two-timestep fixture. It does not change floating-point precision, reduction
order, expected results, or comparison tolerances.

## Completed evidence

`build/rtl_resume_20261025_baseline/lfw_readout_axi/result.json` records a
**Verilog co-simulation PASS**: four transactions, bit-exact score checks,
TLAST on score 53, and KEEP/STRB checks. The first transaction clears state,
the second retains it, and the same two-timestep image is replayed after a
fresh clear. This is two replays of one image, not two distinct images.
The completed result, including the native co-simulation report, is retained
outside ignored build products in `rtl_readout_baseline_result.json`.

The readout campaign completed in 281.337 seconds, with zero OPMODE warnings
and zero tool errors. Its measured latency was 891,795–931,116 cycles,
with 3,645,830 total execution cycles. Random stalls were disabled for this
baseline. HLS estimated clock period: 14.600 ns; this is not routed timing.

## Work in progress

- `rtl_resume_20261025_baseline`: conv1 and conv2 RTL replay is pending.
  Conv1 has completed its first of four transactions; no full conv1 pass
  is claimed until replay and output comparison finish.
- `rtl_resume_20261025_conv_stalls`: conv1/conv2 random-stall replay is pending.
- `rtl_resume_20261025_chain`: integrated finite-FIFO chain replay is pending.

## Random-stall simulator failure

`rtl_resume_20261025_readout_stalls_results.json` records a failed campaign
after 616.265 seconds. XSIM reported `Simulator command interrupted` and
`Simulation engine not responding`, then exited unexpectedly. No RTL
transaction completed and no output mismatch was reported. The generated
`xsimcrash.log` is empty. There were zero OPMODE warnings. This is a failed
verification attempt with an unresolved simulator failure, not an RTL pass
or an established arithmetic defect. The unmodified generated UVM agents
choose random stream and AXI delays bounded at 256 cycles per transfer.

The earlier `rtl_resume_native_20261025` processes ended without campaign
results. Their partial logs are preserved and are not passing evidence.
The sandbox-only `rtl_resume_20261025` attempt failed to open a required
local Vitis port and was interrupted; native runs require local IPC access.

## Limits

A copied conv1 simulation snapshot was replayed for 1 ms with 100 us Tcl
checkpoints (`build/rtl_conv1_time_probe/verilog/time_probe.log`). Simulation
time advanced normally; the probe used 48.390 CPU seconds. Re-elaborating the
same RTL with native XSIM O3 used 43.810 CPU seconds for the same 1 ms
(`time_probe_o3.log`). These bounded diagnostic runs terminate before any
transaction completes and provide no functional pass evidence. They modify
neither DUT RTL nor vendor floating-point models. The measured rate indicates
that full convolution transactions can take tens of minutes or longer.

An RTL match to this fixture does not resolve the separately documented strict
PyTorch numerical divergence. Routed WNS, board inference, full-dataset RTL
accuracy, and distinct-image reset testing are not established by this result.
See `LFW_REPORT.md` for the numerical acceptance gate.
