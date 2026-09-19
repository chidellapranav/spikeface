#include "conv2_pool_axi.h"

// SpikeFace - Checkpoint 3: AXI-ready version of conv2's own pooling
// stage, standalone IP block sitting between conv2_layer_axi and fc1.
// Identical structure to conv1_pool_axi.cpp: buffer the incoming frame
// into an internal array first (pooling needs 4 elements at once, but
// hls::stream only yields one sequential value per read()), then pool
// using the same 4-way partitioning fix already verified.
void conv2_pool_axi(
    hls::stream<spike_t> &input_frame,
    hls::stream<pooled_t> &output_frame
) {
#pragma HLS INTERFACE axis port=input_frame
#pragma HLS INTERFACE axis port=output_frame
#pragma HLS INTERFACE ap_ctrl_hs port=return

    // No persistent state needed -- pooling is a pure function of the
    // current frame, nothing carries over between timesteps.
    spike_t buf[POOL2_CHANNELS][POOL2_IN_H][POOL2_IN_W];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=3

READ_STREAM:
    for (int ch = 0; ch < POOL2_CHANNELS; ch++) {
        for (int r = 0; r < POOL2_IN_H; r++) {
            for (int c = 0; c < POOL2_IN_W; c++) {
#pragma HLS PIPELINE II=1
                buf[ch][r][c] = input_frame.read();
            }
        }
    }

CH_LOOP:
    for (int ch = 0; ch < POOL2_CHANNELS; ch++) {
    ROW_LOOP:
        for (int oy = 0; oy < POOL2_OUT_H; oy++) {
        COL_LOOP:
            for (int ox = 0; ox < POOL2_OUT_W; ox++) {
#pragma HLS PIPELINE II=1
                ap_ufixed<8, 4> sum4 = 0;
                sum4 += (ap_ufixed<8,4>)buf[ch][oy*2][ox*2];
                sum4 += (ap_ufixed<8,4>)buf[ch][oy*2][ox*2+1];
                sum4 += (ap_ufixed<8,4>)buf[ch][oy*2+1][ox*2];
                sum4 += (ap_ufixed<8,4>)buf[ch][oy*2+1][ox*2+1];

                ap_ufixed<8,4> avg_wide = sum4 * ap_ufixed<8,4>(0.25);
                output_frame.write((pooled_t)avg_wide);
            }
        }
    }
}
