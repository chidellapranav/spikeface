#include "conv2_layer_axi.h"

// SpikeFace - Checkpoint 3: AXI-ready conv2 layer.
//
// Interface changes from conv2_layer_step (the verified plain-array
// version, CONV2_IC_GROUP=1, timing-clean at 26 DSP / 9,368 LUT):
//  - input_frame / spike_out: hls::stream, AXI-Stream ports. A whole
//    8x16x16 pooled frame is streamed in (channel-major, raster order --
//    matching conv1_pool_axi's output), and the 16x16x16 output spike map
//    is streamed out the same way per timestep.
//  - weights, beta, threshold: AXI-Lite (control-register) ports, set
//    once by the PS, not re-sent every timestep.
//  - mem_state: REMOVED from the argument list. Now `static` inside the
//    function -- genuine internal, persistent BRAM, same fix already
//    applied and verified for conv1_layer_axi. Deliberately NOT
//    array-partitioned (that caused an II violation on conv1's mem_state
//    -- same lesson applies here).
//  - total_active_taps: kept as an AXI-Lite readable register for
//    sparsity characterization/debug.
//
// The actual computation (padding, per-channel event-gated MAC via
// CONV2_IC_GROUP grouping, LIF update) is UNCHANGED from the verified
// plain-array version -- only the interface wrapping differs.
void conv2_layer_axi(
    hls::stream<pooled_t> &input_frame,
    const weight_t weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][CONV2_K][CONV2_K],
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

    // Persistent membrane potential for all 4096 neurons -- internal to
    // this IP, survives across calls (timesteps) via `static`.
    static mem_t mem_state[CONV2_OUT_CHANNELS][CONV2_H][CONV2_W];

    // Buffer the incoming 8x16x16 pooled frame from the stream.
    pooled_t raw[CONV2_IN_CHANNELS][CONV2_H][CONV2_W];
READ_STREAM:
    for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++) {
        for (int r = 0; r < CONV2_H; r++) {
            for (int c = 0; c < CONV2_W; c++) {
#pragma HLS PIPELINE II=1
                raw[ic][r][c] = input_frame.read();
            }
        }
    }

    // Zero-padded version of each of the 8 input channels (identical to
    // conv2_layer_step's plain-array version, just reading from `raw`
    // instead of directly from a function-argument array).
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
                padded[ic][r][c] = in_bounds ? raw[ic][r - CONV2_PAD][c - CONV2_PAD] : pooled_t(0);
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
            for (int ox = 0; ox < CONV2_W; ox++) {
                pooled_t patch[CONV2_IN_CHANNELS][25];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=0
                for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++)
                    for (int kr = 0; kr < CONV2_K; kr++)
                        for (int kc = 0; kc < CONV2_K; kc++)
                            patch[ic][kr*CONV2_K + kc] = padded[ic][oy + kr][ox + kc];

                acc_t total_cur = 0;
                int active_taps = 0;

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
                spike_out.write(spk);
                total_active_taps += active_taps;
            }
        }
    }
}
