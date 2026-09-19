// SpikeFace - Checkpoint 3: testbench for the full conv1 layer HLS design
// (conv1_layer.h), verified against a full-layer golden reference (8
// channels x 32x32, 20 timesteps) exported bit-exact from the Python
// fixed-point model. This is the array-scale extension of the single-neuron
// verification done in tb_spikeface_neuron.cpp.
#include <cstdio>
#include "conv1_layer.h"
#include "golden_conv1_array.h"

int main() {
    static weight_t weights[CONV1_OUT_CHANNELS][CONV1_K][CONV1_K];
    for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++)
        for (int kr = 0; kr < CONV1_K; kr++)
            for (int kc = 0; kc < CONV1_K; kc++)
                weights[ch][kr][kc] = (weight_t)GOLDEN_ARR_KERNELS[ch][kr][kc];

    static mem_t mem_state[CONV1_OUT_CHANNELS][CONV1_H][CONV1_W];
    for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++)
        for (int y = 0; y < CONV1_H; y++)
            for (int x = 0; x < CONV1_W; x++)
                mem_state[ch][y][x] = 0;

    mem_t beta = (mem_t)GOLDEN_ARR_BETA_Q;
    mem_t threshold = (mem_t)GOLDEN_ARR_THR_Q;

    static spike_t input_frame[CONV1_H][CONV1_W];
    static spike_t spike_out[CONV1_OUT_CHANNELS][CONV1_H][CONV1_W];

    long total_mismatches = 0;
    long grand_total_active_taps = 0;
    long grand_total_possible_taps = (long)GOLDEN_ARR_T * CONV1_OUT_CHANNELS * CONV1_H * CONV1_W * 25;

    printf("%-4s | %-12s | %-16s | %-16s | %-10s\n", "t", "spikes(HLS)", "spikes(golden)", "active_taps", "match?");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_ARR_T; t++) {
        for (int y = 0; y < CONV1_H; y++)
            for (int x = 0; x < CONV1_W; x++)
                input_frame[y][x] = (spike_t)GOLDEN_ARR_INPUT[t][y][x];

        long active_taps = 0;
        conv1_layer_step(input_frame, weights, mem_state, spike_out, beta, threshold, active_taps);
        grand_total_active_taps += active_taps;

        long hls_spike_count = 0;
        long mismatches_this_t = 0;
        for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++) {
            for (int y = 0; y < CONV1_H; y++) {
                for (int x = 0; x < CONV1_W; x++) {
                    int got = (spike_out[ch][y][x] != spike_t(0)) ? 1 : 0;
                    int expected = (int)GOLDEN_ARR_SPIKES[t][ch][y][x];
                    hls_spike_count += got;
                    if (got != expected) mismatches_this_t++;
                }
            }
        }
        total_mismatches += mismatches_this_t;

        long golden_spike_count = 0;
        for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++)
            for (int y = 0; y < CONV1_H; y++)
                for (int x = 0; x < CONV1_W; x++)
                    golden_spike_count += (long)GOLDEN_ARR_SPIKES[t][ch][y][x];

        printf("%-4d | %-12ld | %-16ld | %-16ld | %-10s\n",
               t, hls_spike_count, golden_spike_count, active_taps,
               (mismatches_this_t == 0) ? "OK" : "MISMATCH");
    }

    double sparsity_pct = 100.0 * (1.0 - (double)grand_total_active_taps / grand_total_possible_taps);

    printf("\n--------------------------------------------------------------------\n");
    printf("Total neuron-timesteps checked: %d x %d = %ld\n",
           GOLDEN_ARR_T, CONV1_OUT_CHANNELS * CONV1_H * CONV1_W,
           (long)GOLDEN_ARR_T * CONV1_OUT_CHANNELS * CONV1_H * CONV1_W);
    printf("Total spike mismatches: %ld\n", total_mismatches);
    printf("Total MAC taps possible: %ld, actually computed: %ld\n",
           grand_total_possible_taps, grand_total_active_taps);
    printf("Event-gating sparsity (taps skipped), full layer: %.1f%%\n", sparsity_pct);

    if (total_mismatches == 0) {
        printf("\nPASS: full conv1 array HLS design is bit-exact with the Python fixed-point reference.\n");
        return 0;
    } else {
        printf("\nFAIL: %ld neuron-timestep mismatches across the full layer.\n", total_mismatches);
        return 1;
    }
}
