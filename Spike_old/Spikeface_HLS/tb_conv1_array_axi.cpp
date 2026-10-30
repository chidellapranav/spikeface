// SpikeFace - Checkpoint 3: testbench for conv1_layer_axi (the AXI-ready
// version). Uses the SAME golden vectors as tb_conv1_array.cpp -- if this
// passes with identical results, it confirms the AXI interface wrapping
// (streams in/out, static internal mem_state) didn't change the underlying
// computation at all, only how data enters/exits the block.
#include <cstdio>
#include "conv1_layer_axi.h"
#include "golden_conv1_array.h"

int main() {
    static weight_t weights[CONV1_OUT_CHANNELS][CONV1_K][CONV1_K];
    for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++)
        for (int kr = 0; kr < CONV1_K; kr++)
            for (int kc = 0; kc < CONV1_K; kc++)
                weights[ch][kr][kc] = (weight_t)GOLDEN_ARR_KERNELS[ch][kr][kc];

    mem_t beta = (mem_t)GOLDEN_ARR_BETA_Q;
    mem_t threshold = (mem_t)GOLDEN_ARR_THR_Q;

    long total_mismatches = 0;
    long grand_total_active_taps = 0;
    long grand_total_possible_taps = (long)GOLDEN_ARR_T * CONV1_OUT_CHANNELS * CONV1_H * CONV1_W * 25;

    printf("%-4s | %-12s | %-16s | %-16s | %-10s\n", "t", "spikes(HLS)", "spikes(golden)", "active_taps", "match?");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_ARR_T; t++) {
        hls::stream<spike_t> input_frame;
        hls::stream<spike_t> spike_out;

        // Push the 32x32 input frame for this timestep, in raster order --
        // exactly matching what conv1_layer_axi's READ_STREAM loop expects.
        for (int y = 0; y < CONV1_H; y++)
            for (int x = 0; x < CONV1_W; x++)
                input_frame.write((spike_t)GOLDEN_ARR_INPUT[t][y][x]);

        long active_taps = 0;
        conv1_layer_axi(input_frame, weights, spike_out, beta, threshold, active_taps);
        grand_total_active_taps += active_taps;

        // Pop the 8x32x32 output spikes, in the same channel-major/raster
        // order they were written, and compare against golden.
        long hls_spike_count = 0;
        long golden_spike_count = 0;
        long mismatches_this_t = 0;
        for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++) {
            for (int y = 0; y < CONV1_H; y++) {
                for (int x = 0; x < CONV1_W; x++) {
                    int got = (spike_out.read() != spike_t(0)) ? 1 : 0;
                    int expected = (int)GOLDEN_ARR_SPIKES[t][ch][y][x];
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

    double sparsity_pct = 100.0 * (1.0 - (double)grand_total_active_taps / grand_total_possible_taps);

    printf("\n--------------------------------------------------------------------\n");
    printf("Total neuron-timesteps checked: %ld\n",
           (long)GOLDEN_ARR_T * CONV1_OUT_CHANNELS * CONV1_H * CONV1_W);
    printf("Total spike mismatches: %ld\n", total_mismatches);
    printf("Event-gating sparsity (taps skipped): %.1f%%\n", sparsity_pct);

    if (total_mismatches == 0) {
        printf("\nPASS: AXI-annotated conv1_layer_axi is bit-exact with the plain-array\n");
        printf("reference and the Python fixed-point model. Interface wrapping did not\n");
        printf("change the computation -- persistent static mem_state works correctly\n");
        printf("across calls.\n");
        return 0;
    } else {
        printf("\nFAIL: %ld mismatches. Check stream read/write ordering or the padding logic.\n", total_mismatches);
        return 1;
    }
}
