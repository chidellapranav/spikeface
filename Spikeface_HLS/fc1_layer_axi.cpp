#include "fc1_layer_axi.h"

// SpikeFace - Checkpoint 3: AXI-ready fc1 layer.
//
// Interface changes from fc1_layer_step (the verified plain-array version):
//  - input_flat / spike_out: hls::stream, AXI-Stream ports.
//  - weights, beta, threshold: AXI-Lite (control-register) ports.
//  - mem_state: REMOVED from the argument list. Now `static` inside the
//    function -- genuine internal, persistent BRAM, same fix already
//    applied and verified for conv1_layer_axi/conv2_layer_axi.
//  - total_active_taps: kept as an AXI-Lite readable register.
//
// The computation itself (FC1_GROUP-wise time-multiplexed dense MAC, LIF
// update) is UNCHANGED from the verified plain-array version.
void fc1_layer_axi(
    hls::stream<pooled_t> &input_flat,
    const weight_t weights[FC1_OUT][FC1_IN],
    hls::stream<spike_t> &spike_out,
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
) {
#pragma HLS INTERFACE axis port=input_flat
#pragma HLS INTERFACE axis port=spike_out
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=total_active_taps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Persistent membrane potential for all 64 output neurons.
    static mem_t mem_state[FC1_OUT];

    // Buffer the incoming 1024 pooled values from the stream.
    pooled_t buf[FC1_IN];
READ_STREAM:
    for (int i = 0; i < FC1_IN; i++) {
#pragma HLS PIPELINE II=1
        buf[i] = input_flat.read();
    }

    total_active_taps = 0;

OUT_LOOP:
    for (int out_idx = 0; out_idx < FC1_OUT; out_idx++) {
        acc_t total_cur = 0;
        int active_taps = 0;

    GROUP_LOOP:
        for (int g = 0; g < FC1_IN / FC1_GROUP; g++) {
#pragma HLS PIPELINE II=1
        GROUP_INNER:
            for (int gi = 0; gi < FC1_GROUP; gi++) {
#pragma HLS UNROLL
                int idx = g * FC1_GROUP + gi;
                if (buf[idx] != pooled_t(0)) {
                    total_cur += acc_t(buf[idx]) * acc_t(weights[out_idx][idx]);
                    active_taps++;
                }
            }
        }

        mem_t mem_local = mem_state[out_idx];
        spike_t spk = lif_update(total_cur, mem_local, beta, threshold);
        mem_state[out_idx] = mem_local;
        spike_out.write(spk);
        total_active_taps += active_taps;
    }
}
