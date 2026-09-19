#include "conv1_pool.h"

// SpikeFace - Checkpoint 3 (conv2): pooling stage.
//
// Purely a spatial downsample -- NOT a neuron layer. No LIF, no threshold,
// no membrane state. Matches Python's F.avg_pool2d(spk1, 2) exactly: the
// arithmetic mean of each 2x2 block of conv1's binary spike output.
void conv1_pool_step(
    const spike_t input_frame[POOL_CHANNELS][POOL_IN_H][POOL_IN_W],
    pooled_t output_frame[POOL_CHANNELS][POOL_OUT_H][POOL_OUT_W]
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
    // Each pooling iteration reads 4 elements from input_frame in the same
    // cycle (the four corners of a 2x2 block). Without partitioning, the
    // array defaults to 2 read ports, forcing those 4 reads across 2
    // cycles and dropping achieved Interval from 1 to 2 (an "II Violation:
    // Resource Limitation" in synthesis). Partitioning both spatial
    // dimensions cyclically by factor 2 splits input_frame into 4 banks --
    // one per corner of every 2x2 block -- so all 4 reads happen in
    // parallel, restoring II=1.
#pragma HLS ARRAY_PARTITION variable=input_frame cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=input_frame cyclic factor=2 dim=3

CH_LOOP:
    for (int ch = 0; ch < POOL_CHANNELS; ch++) {
    ROW_LOOP:
        for (int oy = 0; oy < POOL_OUT_H; oy++) {
        COL_LOOP:
            for (int ox = 0; ox < POOL_OUT_W; ox++) {
#pragma HLS PIPELINE II=1
                // Sum of 4 binary spikes is an integer in [0,4]. Use a wide
                // enough intermediate type to hold that sum safely before
                // scaling by 0.25 -- pooled_t itself only ranges up to 1.75,
                // so the un-scaled sum (up to 4) would overflow it directly.
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
