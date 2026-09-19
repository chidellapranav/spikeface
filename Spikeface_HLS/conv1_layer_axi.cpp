#include "conv1_layer_axi.h"

void conv1_layer_axi(
    hls::stream<spike_t> &input_frame,
    const weight_t weights[CONV1_OUT_CHANNELS][CONV1_K][CONV1_K],
    hls::stream<spike_t> &spike_out,
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
) {
#pragma HLS INTERFACE axis port=input_frame
#pragma HLS INTERFACE axis port=spike_out
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=total_active_taps bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    static mem_t mem_state[CONV1_OUT_CHANNELS][CONV1_H][CONV1_W];

    spike_t padded[CONV1_H + 2*CONV1_PAD][CONV1_W + 2*CONV1_PAD];
#pragma HLS ARRAY_PARTITION variable=padded cyclic factor=5 dim=2

ZERO_PAD:
    for (int r = 0; r < CONV1_H + 2*CONV1_PAD; r++) {
        for (int c = 0; c < CONV1_W + 2*CONV1_PAD; c++) {
#pragma HLS PIPELINE II=1
            padded[r][c] = spike_t(0);
        }
    }

READ_STREAM:
    for (int r = 0; r < CONV1_H; r++) {
        for (int c = 0; c < CONV1_W; c++) {
#pragma HLS PIPELINE II=1
            padded[r + CONV1_PAD][c + CONV1_PAD] = input_frame.read();
        }
    }

    total_active_taps = 0;

CH_LOOP:
    for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++) {
        weight_t kernel_flat[25];
#pragma HLS ARRAY_PARTITION variable=kernel_flat complete
        for (int kr = 0; kr < CONV1_K; kr++)
            for (int kc = 0; kc < CONV1_K; kc++)
                kernel_flat[kr*CONV1_K + kc] = weights[ch][kr][kc];

    ROW_LOOP:
        for (int oy = 0; oy < CONV1_H; oy++) {
        COL_LOOP:
            for (int ox = 0; ox < CONV1_W; ox++) {
#pragma HLS PIPELINE II=1
                spike_t patch[25];
#pragma HLS ARRAY_PARTITION variable=patch complete
                for (int kr = 0; kr < CONV1_K; kr++)
                    for (int kc = 0; kc < CONV1_K; kc++)
                        patch[kr*CONV1_K + kc] = padded[oy + kr][ox + kc];

                int active_taps = 0;
                mem_t mem_local = mem_state[ch][oy][ox];
                spike_t spk = spikeface_neuron_step(patch, kernel_flat, mem_local,
                                                     beta, threshold, active_taps);
                mem_state[ch][oy][ox] = mem_local;
                spike_out.write(spk);
                total_active_taps += active_taps;
            }
        }
    }
}
