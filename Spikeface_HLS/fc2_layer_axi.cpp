#include "fc2_layer_axi.h"

// SpikeFace - Checkpoint 3: fc2 (readout) layer.
//
// Fully parallel per-neuron MAC (all 64 taps in one cycle) -- safe here
// specifically because the input is spike_t (single-bit), so every "tap"
// is a free conditional-accumulate, not a real multiply (see header
// comment). This is the same efficiency conv1 relies on; conv2/fc1 needed
// FC*_GROUP time-multiplexing precisely because THEIR inputs were
// multi-valued pooled_t, not because of fan-in size alone.
void fc2_layer_axi(
    hls::stream<spike_t> &fc1_spikes_in,
    const weight_fc2_t weights[FC2_OUT][FC2_IN],
    ap_uint<1> clear_accum,
    hls::stream<score_packet_t> &class_scores_out
) {
#pragma HLS INTERFACE axis port=fc1_spikes_in
#pragma HLS INTERFACE axis port=class_scores_out
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=clear_accum bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Persistent accumulator across timesteps -- internal to this IP.
    // Cleared only when the PS asserts clear_accum (start of a new
    // image's inference), not every call.
    static acc_t accum[FC2_OUT];

    // Buffer the incoming 64 spikes.
    spike_t buf[FC2_IN];
READ_STREAM:
    for (int i = 0; i < FC2_IN; i++) {
#pragma HLS PIPELINE II=1
        buf[i] = fc1_spikes_in.read();
    }

OUT_LOOP:
    for (int out_idx = 0; out_idx < FC2_OUT; out_idx++) {
        acc_t cur = 0;
        // Process ONE tap per cycle -- NOT for multiplier/DSP reasons
        // (per the header comment, spike_t input still makes the
        // multiply free), but because `weights` is bound over AXI-Lite,
        // which provides only ONE READ PORT. Every output neuron needs
        // its own unique 64-weight row (fc2 is fully-connected, so no
        // weight sharing/reuse exists to amortize this against, unlike
        // conv1/conv2's kernel reuse). Fetching 64 different weights in
        // one cycle from a single-port memory is structurally impossible
        // regardless of the arithmetic. An earlier version fully unrolled
        // this loop, assuming that was safe since the multiply itself is
        // free -- synthesis showed an explicit "II Violation: Resource
        // Limitation" with achieved Interval=64 (not the requested 1),
        // and 64 DSPs, because the forced 64-cycle serialization from the
        // memory bottleneck disrupted the optimizer's ability to
        // recognize/merge the intended-parallel copies into a single
        // shared free-select structure. Matching fc1's already-proven
        // one-tap-per-cycle pattern (FC1_GROUP=1) resolves this the same
        // way it already resolves it there.
    MAC_LOOP:
        for (int i = 0; i < FC2_IN; i++) {
#pragma HLS PIPELINE II=1
            // Restructured from a conditional accumulate (`if (gate) cur
            // += ...`) to an unconditional add of a pre-selected value.
            // The conditional version passed C-Simulation and resolved
            // the earlier resource-limitation violation, but left a
            // small (-0.21 ns) timing violation -- the conditional
            // accumulator-enable logic added just enough extra
            // combinational depth to miss the 10 ns target. Masking to
            // zero first, then always adding, removes that conditional
            // control path from the adder itself.
            acc_t term = (buf[i] != spike_t(0)) ? acc_t(weights[out_idx][i]) : acc_t(0);
            cur += term;
        }

        acc_t updated;
        if (clear_accum) {
            updated = cur;              // reset: this timestep's contribution only
        } else {
            updated = accum[out_idx] + cur;  // accumulate on top of running total
        }
        accum[out_idx] = updated;       // ap_fixed assignment applies AP_RND_CONV/AP_SAT

        // Pack into an AXI-Stream packet with an explicit TLAST bit.
        // .data holds the RAW 32-bit pattern of the fixed-point score
        // (a bit-pattern copy via .range(), not a numeric cast -- the
        // receiving software reinterprets these same 32 bits back into
        // the equivalent fixed-point value). .last is asserted only on
        // the final (40th) score of this call, signaling one complete
        // DMA transfer per timestep.
        score_packet_t pkt;
        pkt.data = axis_score_t(updated).range(31, 0);
        pkt.keep = -1;  // all 4 bytes valid
        pkt.last = (out_idx == FC2_OUT - 1) ? 1 : 0;
        class_scores_out.write(pkt);
    }
}
