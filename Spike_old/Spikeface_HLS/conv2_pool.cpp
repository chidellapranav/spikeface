#include "conv2_pool.h"

// SpikeFace - Checkpoint 3: conv2's own pooling stage (16ch 16x16 -> 8x8),
// feeding fc1. Purely a spatial downsample, no LIF/threshold/membrane
// state. Identical structure to conv1_pool.cpp -- same partitioning fix
// (4-way bank split so all 4 corner reads happen in the same cycle) is
// applied here too, resized for the new dimensions.
void conv2_pool_step(
    const spike_t input_frame[POOL2_CHANNELS][POOL2_IN_H][POOL2_IN_W],
    pooled_t output_frame[POOL2_CHANNELS][POOL2_OUT_H][POOL2_OUT_W]
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
    // Each pooling iteration reads 4 elements from input_frame in the same
    // cycle (the four corners of a 2x2 block). Partition both spatial
    // dimensions cyclically by factor 2 -- same fix verified for
    // conv1_pool.cpp -- so all 4 reads land in separate banks and II=1
    // is achievable.
#pragma HLS ARRAY_PARTITION variable=input_frame cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=input_frame cyclic factor=2 dim=3

CH_LOOP:
    for (int ch = 0; ch < POOL2_CHANNELS; ch++) {
    ROW_LOOP:
        for (int oy = 0; oy < POOL2_OUT_H; oy++) {
        COL_LOOP:
            for (int ox = 0; ox < POOL2_OUT_W; ox++) {
#pragma HLS PIPELINE II=1
                ap_ufixed<8, 4> sum4 = 0;
                sum4 += (ap_ufixed<8,4>)input_frame[ch][oy*2][ox*2];
                sum4 += (ap_ufixed<8,4>)input_frame[ch][oy*2][ox*2+1];
                sum4 += (ap_ufixed<8,4>)input_frame[ch][oy*2+1][ox*2];
                sum4 += (ap_ufixed<8,4>)input_frame[ch][oy*2+1][ox*2+1];

                ap_ufixed<8,4> avg_wide = sum4 * ap_ufixed<8,4>(0.25);
                output_frame[ch][oy][ox] = (pooled_t)avg_wide;
            }
        }
    }
}
