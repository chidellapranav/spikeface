// SpikeFace - Checkpoint 3: testbench for conv2_layer_axi. Uses the SAME
// golden vectors as tb_conv2_array.cpp -- if this passes with identical
// results, the stream/AXI interface wrapping didn't change the computation.
#include <cstdio>
#include "conv2_layer_axi.h"
#include "golden_conv2.h"

int main() {
    static weight_t weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][CONV2_K][CONV2_K];
    for (int ch = 0; ch < CONV2_OUT_CHANNELS; ch++)
        for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++)
            for (int kr = 0; kr < CONV2_K; kr++)
                for (int kc = 0; kc < CONV2_K; kc++)
                    weights[ch][ic][kr][kc] = (weight_t)GOLDEN_CONV2_WEIGHTS[ch][ic][kr][kc];

    mem_t beta = (mem_t)GOLDEN_CONV2_BETA_Q;
    mem_t threshold = (mem_t)GOLDEN_CONV2_THR_Q;

    long total_mismatches = 0;
    long grand_total_active_taps = 0;

    printf("%-4s | %-12s | %-16s | %-16s | %-10s\n", "t", "spikes(HLS)", "spikes(golden)", "active_taps", "match?");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_CONV2_T; t++) {
        hls::stream<pooled_t> input_frame;
        hls::stream<spike_t> spike_out;

        // Push in channel-major, raster order -- matches conv1_pool_axi's
        // output ordering.
        for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++)
            for (int y = 0; y < CONV2_H; y++)
                for (int x = 0; x < CONV2_W; x++)
                    input_frame.write((pooled_t)GOLDEN_CONV2_INPUT[t][ic][y][x]);

        long active_taps = 0;
        conv2_layer_axi(input_frame, weights, spike_out, beta, threshold, active_taps);
        grand_total_active_taps += active_taps;

        long hls_spike_count = 0, golden_spike_count = 0, mismatches_this_t = 0;
        for (int ch = 0; ch < CONV2_OUT_CHANNELS; ch++) {
            for (int y = 0; y < CONV2_H; y++) {
                for (int x = 0; x < CONV2_W; x++) {
                    int got = (spike_out.read() != spike_t(0)) ? 1 : 0;
                    int expected = (int)GOLDEN_CONV2_SPIKES[t][ch][y][x];
                    hls_spike_count += got;
                    golden_spike_count += expected;
                    if (got != expected) mismatches_this_t++;
                }
            }
        }
        total_mismatches += mismatches_this_t;

        printf("%-4d | %-12ld | %-16ld | %-16ld | %-10s\n",
               t, hls_spike_count, golden_spike_count, active_taps,
               (mismatches_this_t == 0) ? "OK" : "MISMATCH");
    }

    long total_possible = (long)GOLDEN_CONV2_T * CONV2_OUT_CHANNELS * CONV2_H * CONV2_W * 200;
    double sparsity_pct = 100.0 * (1.0 - (double)grand_total_active_taps / total_possible);

    printf("\n--------------------------------------------------------------------\n");
    printf("Total neuron-timesteps checked: %ld\n",
           (long)GOLDEN_CONV2_T * CONV2_OUT_CHANNELS * CONV2_H * CONV2_W);
    printf("Total spike mismatches: %ld\n", total_mismatches);
    printf("Event-gating sparsity (taps skipped): %.1f%%\n", sparsity_pct);

    if (total_mismatches == 0) {
        printf("\nPASS: AXI-ready conv2 layer is bit-exact with the plain-array\n");
        printf("reference and the Python fixed-point model.\n");
        return 0;
    } else {
        printf("\nFAIL: %ld mismatches.\n", total_mismatches);
        return 1;
    }
}
