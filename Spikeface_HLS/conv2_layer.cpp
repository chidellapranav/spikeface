#include "conv2_layer.h"

// SpikeFace - Checkpoint 3: conv2 layer (16 output channels x 16x16 spatial
// = 4096 neurons, 8-channel fan-in), one timestep.
//
// TUNABLE PARALLELISM: CONV2_IC_GROUP (defined in conv2_layer.h) controls
// how many of the 8 input channels are processed simultaneously per cycle.
//   - CONV2_IC_GROUP=1 : fully time-multiplexed (~26 DSP, ~9.4K LUT,
//     ~1.03M cycles/timestep -- the "safe but slow" version, verified on
//     real hardware synthesis).
//   - CONV2_IC_GROUP=8 : fully parallel (~201 DSP, ~118K LUT -- does NOT
//     fit on xc7z020; this is the version that first blew the budget).
//   - Values in between (2 or 4) trade DSP/LUT for latency: roughly
//     CONV2_IC_GROUP x the DSP count of the group=1 version, and roughly
//     1/CONV2_IC_GROUP x its channel-loop latency. Must evenly divide 8
//     (CONV2_IN_CHANNELS) -- valid choices are 1, 2, 4, 8.
//
// Root cause history: PIPELINE II=1 was originally on ROW_LOOP/COL_LOOP,
// forcing all 8 channels' 25-tap MACs to complete in one cycle (200 real
// multipliers -- pooled_t values are genuine fractions, not free
// single-bit multiplies like conv1's spike_t). An ALLOCATION pragma was
// tried to cap multiplier count directly -- had ZERO effect (identical
// synthesis report). Fixed by moving PIPELINE down to the (now grouped)
// channel loop, so only CONV2_IC_GROUP channels' worth of real multiplier
// hardware exists, reused across the remaining groups sequentially.
void conv2_layer_step(
    const pooled_t input_frame[CONV2_IN_CHANNELS][CONV2_H][CONV2_W],
    const weight_t weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][CONV2_K][CONV2_K],
    mem_t mem_state[CONV2_OUT_CHANNELS][CONV2_H][CONV2_W],
    spike_t spike_out[CONV2_OUT_CHANNELS][CONV2_H][CONV2_W],
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return

    // Zero-padded version of each of the 8 input channels.
    pooled_t padded[CONV2_IN_CHANNELS][CONV2_H + 2*CONV2_PAD][CONV2_W + 2*CONV2_PAD];

PAD_IC:
    for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++) {
    PAD_ROWS:
        for (int r = 0; r < CONV2_H + 2*CONV2_PAD; r++) {
        PAD_COLS:
            for (int c = 0; c < CONV2_W + 2*CONV2_PAD; c++) {
#pragma HLS PIPELINE II=1
                bool in_bounds = (r >= CONV2_PAD) && (r < CONV2_H + CONV2_PAD) &&
                                  (c >= CONV2_PAD) && (c < CONV2_W + CONV2_PAD);
                padded[ic][r][c] = in_bounds ? input_frame[ic][r - CONV2_PAD][c - CONV2_PAD] : pooled_t(0);
            }
        }
    }

    total_active_taps = 0;

CH_LOOP:
    for (int ch = 0; ch < CONV2_OUT_CHANNELS; ch++) {
        weight_t kernel_flat[CONV2_IN_CHANNELS][25];
#pragma HLS ARRAY_PARTITION variable=kernel_flat complete dim=0
        for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++)
            for (int kr = 0; kr < CONV2_K; kr++)
                for (int kc = 0; kc < CONV2_K; kc++)
                    kernel_flat[ic][kr*CONV2_K + kc] = weights[ch][ic][kr][kc];

    ROW_LOOP:
        for (int oy = 0; oy < CONV2_H; oy++) {
        COL_LOOP:
            // No PIPELINE pragma here -- pushed down to IC_GROUP_LOOP below.
            for (int ox = 0; ox < CONV2_W; ox++) {
                pooled_t patch[CONV2_IN_CHANNELS][25];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=0
                for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++)
                    for (int kr = 0; kr < CONV2_K; kr++)
                        for (int kc = 0; kc < CONV2_K; kc++)
                            patch[ic][kr*CONV2_K + kc] = padded[ic][oy + kr][ox + kc];

                acc_t total_cur = 0;
                int active_taps = 0;

                // Process CONV2_IC_GROUP channels per cycle (unrolled),
                // looping over (8 / CONV2_IC_GROUP) groups sequentially
                // (pipelined at II=1 across groups).
            IC_GROUP_LOOP:
                for (int g = 0; g < CONV2_IN_CHANNELS / CONV2_IC_GROUP; g++) {
#pragma HLS PIPELINE II=1
                IC_INNER:
                    for (int gi = 0; gi < CONV2_IC_GROUP; gi++) {
#pragma HLS UNROLL
                        int ic = g * CONV2_IC_GROUP + gi;
                        int taps_ic = 0;
                        acc_t cur = event_gated_mac25_pooled(patch[ic], kernel_flat[ic], taps_ic);
                        total_cur += cur;
                        active_taps += taps_ic;
                    }
                }

                mem_t mem_local = mem_state[ch][oy][ox];
                spike_t spk = lif_update(total_cur, mem_local, beta, threshold);
                mem_state[ch][oy][ox] = mem_local;
                spike_out[ch][oy][ox] = spk;
                total_active_taps += active_taps;
            }
        }
    }
}
