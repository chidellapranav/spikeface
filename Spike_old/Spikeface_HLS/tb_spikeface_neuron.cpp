// SpikeFace - Checkpoint 2: standalone testbench for the LIF neuron core.
//
// Compares the HLS C++ neuron (spikeface_neuron_step, using ap_fixed types)
// against golden vectors exported bit-exact from the Python quantized
// reference model (quantization_sweep.py). This runs as a plain g++ csim
// using Xilinx's open-sourced ap_fixed headers, since full Vitis HLS is not
// available in this environment -- but this is exactly the C-simulation step
// Vitis HLS itself runs before synthesis, so a pass here means the design is
// ready to drop into a real Vitis HLS project for synthesis/RTL export.
#include <cstdio>
#include <cmath>
#include "spikeface_neuron_top.h"
#include "golden_vectors.h"

int main() {
    weight_t kernel[25];
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 5; c++)
            kernel[r*5+c] = (weight_t)GOLDEN_KERNEL[r][c];

    mem_t beta = (mem_t)GOLDEN_BETA_Q;
    mem_t threshold = (mem_t)GOLDEN_THR_Q;
    mem_t mem = 0;

    int mismatches = 0;
    int total_active_taps = 0;
    int total_possible_taps = GOLDEN_T * 25;

    printf("%-4s | %-8s | %-14s | %-14s | %-10s\n", "t", "spike", "expected_spike", "active_taps", "match?");
    printf("---------------------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_T; t++) {
        spike_t patch[25];
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++)
                patch[r*5+c] = (spike_t)GOLDEN_PATCHES[t][r][c];

        int active_taps = 0;
        spike_t spk = spikeface_neuron_step(patch, kernel, mem, beta, threshold, active_taps);

        int expected_spike = (int)GOLDEN_SPIKES[t];
        int got_spike = (spk != spike_t(0)) ? 1 : 0;
        bool match = (got_spike == expected_spike);
        if (!match) mismatches++;
        total_active_taps += active_taps;

        printf("%-4d | %-8d | %-14d | %-14d | %-10s\n",
               t, got_spike, expected_spike, active_taps, match ? "OK" : "MISMATCH");
    }

    double sparsity_pct = 100.0 * (1.0 - (double)total_active_taps / total_possible_taps);

    printf("\n---------------------------------------------------------------\n");
    printf("Total timesteps: %d\n", GOLDEN_T);
    printf("Mismatches: %d\n", mismatches);
    printf("Total MAC taps possible: %d, actually computed: %d\n", total_possible_taps, total_active_taps);
    printf("Event-gating sparsity (taps skipped): %.1f%%\n", sparsity_pct);

    if (mismatches == 0) {
        printf("\nPASS: HLS neuron core is bit-exact with the Python fixed-point reference.\n");
        return 0;
    } else {
        printf("\nFAIL: %d/%d timesteps mismatched.\n", mismatches, GOLDEN_T);
        return 1;
    }
}
