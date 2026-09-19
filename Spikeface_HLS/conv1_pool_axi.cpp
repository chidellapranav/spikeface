#include "conv1_pool_axi.h"

// SpikeFace - Checkpoint 3: AXI-ready pooling stage, standalone IP block
// sitting between conv1_layer_axi and conv2_layer_axi.
//
// Pooling needs 4 elements at once per output (the 2x2 block corners), but
// hls::stream only yields one sequential value per read() -- and the 4
// corners aren't adjacent in raster order (two of them are a full row
// apart). So the incoming frame is buffered into an internal array first,
// then pooled using the same array-partitioning fix already verified in
// the plain-array version (conv1_pool.cpp) -- partitioning both spatial
// dimensions by factor 2 gives 4 independent banks, so all 4 reads happen
// in the same cycle and II=1 is achievable, exactly as confirmed there.
void conv1_pool_axi(
    hls::stream<spike_t> &input_frame,
    hls::stream<pooled_t> &output_frame
) {
#pragma HLS INTERFACE axis port=input_frame
#pragma HLS INTERFACE axis port=output_frame
#pragma HLS INTERFACE ap_ctrl_hs port=return

    // No persistent state needed -- pooling is a pure function of the
    // current frame, nothing carries over between timesteps. Plain local
    // array (not `static`), unlike mem_state in conv1_layer_axi/
    // conv2_layer_axi.
    spike_t buf[POOL_CHANNELS][POOL_IN_H][POOL_IN_W];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=2 dim=3

READ_STREAM:
    for (int ch = 0; ch < POOL_CHANNELS; ch++) {
        for (int r = 0; r < POOL_IN_H; r++) {
            for (int c = 0; c < POOL_IN_W; c++) {
#pragma HLS PIPELINE II=1
                buf[ch][r][c] = input_frame.read();
            }
        }
    }

CH_LOOP:
    for (int ch = 0; ch < POOL_CHANNELS; ch++) {
    ROW_LOOP:
        for (int oy = 0; oy < POOL_OUT_H; oy++) {
        COL_LOOP:
            for (int ox = 0; ox < POOL_OUT_W; ox++) {
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
