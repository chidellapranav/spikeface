// SpikeFace - Checkpoint 3: full conv1 layer (8 output channels x 32x32 spatial
// = 8192 neurons), one timestep.
//
// Design decisions carried over from the Checkpoint 2 synthesis findings
// (see CHECKPOINT2_STATUS.md):
//  - The verified single-neuron core (event_gated_mac25 + lif_update) is
//    TIME-MULTIPLEXED across all 8192 neurons rather than physically
//    replicated -- at ~4,658 LUT/neuron (8% of xc7z020 for ONE neuron),
//    8192 physical copies is not remotely feasible.
//  - Membrane potential for all 8192 neurons is held in an on-chip array
//    (mem_state[8][32][32]), which Vitis HLS will map to BRAM rather than
//    flip-flops, since it's addressed by loop variables rather than fully
//    unrolled.
//  - The neuron-update loop is pipelined (target II=1, matching the fix
//    applied to spikeface_neuron_top.h) so successive neuron updates
//    overlap instead of fully serializing.
//  - Weights are loaded once per layer (8 output channels x 25 taps = 200
//    weights total for conv1) and reused across all 32x32 spatial positions
//    of a given output channel -- weight reuse, not replication, matching
//    how a real conv layer shares kernel weights spatially.
#ifndef CONV1_LAYER_H
#define CONV1_LAYER_H

#include "spikeface_types.h"
#include "lif_neuron.h"
#include "spikeface_neuron_top.h"

#define CONV1_OUT_CHANNELS 8
#define CONV1_H 32
#define CONV1_W 32
#define CONV1_K 5
#define CONV1_PAD 2

// One timestep of the full conv1 layer.
//
// input_frame:  32x32 single-channel input spike frame for this timestep
//               (already binarized 0/1, e.g. from the rate-coded encoder).
// weights:      [8][5][5] conv1 kernel weights (loaded once, reused spatially).
// mem_state:    [8][32][32] persistent membrane potential, in/out (BRAM).
// spike_out:    [8][32][32] this timestep's output spike map.
// beta, threshold: shared neuron parameters (same for all neurons in this design;
//               a per-channel threshold would be a straightforward extension).
// total_active_taps: running count of taps actually computed (for sparsity reporting).
void conv1_layer_step(
    const spike_t input_frame[CONV1_H][CONV1_W],
    const weight_t weights[CONV1_OUT_CHANNELS][CONV1_K][CONV1_K],
    mem_t mem_state[CONV1_OUT_CHANNELS][CONV1_H][CONV1_W],
    spike_t spike_out[CONV1_OUT_CHANNELS][CONV1_H][CONV1_W],
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
) {
#pragma HLS INLINE off

    // Zero-padded input, built once per timestep and reused across all 8
    // output channels (avoids recomputing the pad for every channel).
    spike_t padded[CONV1_H + 2*CONV1_PAD][CONV1_W + 2*CONV1_PAD];
#pragma HLS ARRAY_PARTITION variable=padded cyclic factor=5 dim=2

PAD_ROWS:
    for (int r = 0; r < CONV1_H + 2*CONV1_PAD; r++) {
PAD_COLS:
        for (int c = 0; c < CONV1_W + 2*CONV1_PAD; c++) {
#pragma HLS PIPELINE II=1
            bool in_bounds = (r >= CONV1_PAD) && (r < CONV1_H + CONV1_PAD) &&
                              (c >= CONV1_PAD) && (c < CONV1_W + CONV1_PAD);
            padded[r][c] = in_bounds ? input_frame[r - CONV1_PAD][c - CONV1_PAD] : spike_t(0);
        }
    }

    total_active_taps = 0;

CH_LOOP:
    for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++) {
        // Flatten this channel's 5x5 kernel once per channel (reused for all
        // 1024 spatial positions in that channel -- weight reuse, matching
        // real conv-layer semantics).
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
                spike_out[ch][oy][ox] = spk;
                total_active_taps += active_taps;
            }
        }
    }
}

#endif
