#include "fc1_layer.h"

// SpikeFace - Checkpoint 3: fc1 layer (fully-connected, 1024 -> 64,
// spiking), one timestep.
//
// Structurally different from conv1/conv2: no spatial patches, no
// padding, no kernels -- every one of the 1024 pooled inputs connects to
// every one of 64 output neurons (a dense/all-to-all layer). Applies the
// same grouped time-multiplexing pattern conv2 needed (see conv2_layer.cpp
// for the full trade-off history): FC1_GROUP controls how many of the
// 1024 inputs are summed in parallel per cycle, looping over
// (1024/FC1_GROUP) groups sequentially. Starting directly at FC1_GROUP=1
// (the verified-safe setting from conv2) rather than discovering the same
// LUT/DSP blowup by trying full parallelism first.
void fc1_layer_step(
    const pooled_t input_flat[FC1_IN],
    const weight_t weights[FC1_OUT][FC1_IN],
    mem_t mem_state[FC1_OUT],
    spike_t spike_out[FC1_OUT],
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return

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
                if (input_flat[idx] != pooled_t(0)) {
                    total_cur += acc_t(input_flat[idx]) * acc_t(weights[out_idx][idx]);
                    active_taps++;
                }
            }
        }

        mem_t mem_local = mem_state[out_idx];
        spike_t spk = lif_update(total_cur, mem_local, beta, threshold);
        mem_state[out_idx] = mem_local;
        spike_out[out_idx] = spk;
        total_active_taps += active_taps;
    }
}
