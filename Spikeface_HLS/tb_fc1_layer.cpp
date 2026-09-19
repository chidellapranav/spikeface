#include <cstdio>
#include "fc1_layer.h"
#include "golden_fc1.h"

int main() {
    static weight_t weights[FC1_OUT][FC1_IN];
    for (int o = 0; o < FC1_OUT; o++)
        for (int i = 0; i < FC1_IN; i++)
            weights[o][i] = (weight_t)GOLDEN_FC1_WEIGHTS[o][i];

    static mem_t mem_state[FC1_OUT];
    for (int o = 0; o < FC1_OUT; o++) mem_state[o] = 0;

    mem_t beta = (mem_t)GOLDEN_FC1_BETA_Q;
    mem_t threshold = (mem_t)GOLDEN_FC1_THR_Q;

    static pooled_t input_flat[FC1_IN];
    static spike_t spike_out[FC1_OUT];

    long total_mismatches = 0;
    long grand_total_active_taps = 0;

    printf("%-4s | %-12s | %-16s | %-16s | %-10s\n", "t", "spikes(HLS)", "spikes(golden)", "active_taps", "match?");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_FC1_T; t++) {
        for (int i = 0; i < FC1_IN; i++)
            input_flat[i] = (pooled_t)GOLDEN_FC1_INPUT[t][i];

        long active_taps = 0;
        fc1_layer_step(input_flat, weights, mem_state, spike_out, beta, threshold, active_taps);
        grand_total_active_taps += active_taps;

        long hls_spike_count = 0, golden_spike_count = 0, mismatches_this_t = 0;
        for (int o = 0; o < FC1_OUT; o++) {
            int got = (spike_out[o] != spike_t(0)) ? 1 : 0;
            int expected = (int)GOLDEN_FC1_SPIKES[t][o];
            hls_spike_count += got;
            golden_spike_count += expected;
            if (got != expected) mismatches_this_t++;
        }
        total_mismatches += mismatches_this_t;

        printf("%-4d | %-12ld | %-16ld | %-16ld | %-10s\n",
               t, hls_spike_count, golden_spike_count, active_taps,
               (mismatches_this_t == 0) ? "OK" : "MISMATCH");
    }

    long total_possible = (long)GOLDEN_FC1_T * FC1_OUT * FC1_IN;
    double sparsity_pct = 100.0 * (1.0 - (double)grand_total_active_taps / total_possible);

    printf("\n--------------------------------------------------------------------\n");
    printf("Total neuron-timesteps checked: %ld\n", (long)GOLDEN_FC1_T * FC1_OUT);
    printf("Total spike mismatches: %ld\n", total_mismatches);
    printf("Total MAC taps possible: %ld, actually computed: %ld\n", total_possible, grand_total_active_taps);
    printf("Event-gating sparsity (taps skipped): %.1f%%\n", sparsity_pct);

    if (total_mismatches == 0) {
        printf("\nPASS: fc1 layer is bit-exact with the Python fixed-point reference.\n");
        return 0;
    } else {
        printf("\nFAIL: %ld mismatches.\n", total_mismatches);
        return 1;
    }
}
