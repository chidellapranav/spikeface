// SpikeFace - Checkpoint 3: testbench for conv1_pool_axi (AXI-ready
// pooling). Uses the SAME golden vectors as tb_conv1_pool.cpp -- if this
// passes with identical results, the stream interface wrapping didn't
// change the computation.
#include <cstdio>
#include "conv1_pool_axi.h"
#include "golden_pool.h"

int main() {
    long total_mismatches = 0;
    double max_abs_err = 0.0;

    for (int t = 0; t < GOLDEN_POOL_T; t++) {
        hls::stream<spike_t> input_frame;
        hls::stream<pooled_t> output_frame;

        // Push in channel-major, raster order -- matches how
        // conv1_layer_axi's spike_out stream is written.
        for (int ch = 0; ch < POOL_CHANNELS; ch++)
            for (int y = 0; y < POOL_IN_H; y++)
                for (int x = 0; x < POOL_IN_W; x++)
                    input_frame.write((spike_t)GOLDEN_POOL_INPUT[t][ch][y][x]);

        conv1_pool_axi(input_frame, output_frame);

        for (int ch = 0; ch < POOL_CHANNELS; ch++) {
            for (int y = 0; y < POOL_OUT_H; y++) {
                for (int x = 0; x < POOL_OUT_W; x++) {
                    double got = (double)output_frame.read();
                    double expected = GOLDEN_POOL_OUTPUT[t][ch][y][x];
                    double err = got - expected;
                    if (err < 0) err = -err;
                    if (err > max_abs_err) max_abs_err = err;
                    if (err > 1e-9) {
                        total_mismatches++;
                        if (total_mismatches <= 10) {
                            printf("MISMATCH t=%d ch=%d y=%d x=%d got=%f expected=%f\n",
                                   t, ch, y, x, got, expected);
                        }
                    }
                }
            }
        }
    }

    printf("Total values checked: %d x %d = %ld\n",
           GOLDEN_POOL_T, POOL_CHANNELS*POOL_OUT_H*POOL_OUT_W,
           (long)GOLDEN_POOL_T*POOL_CHANNELS*POOL_OUT_H*POOL_OUT_W);
    printf("Mismatches: %ld\n", total_mismatches);
    printf("Max abs error: %.10f\n", max_abs_err);

    if (total_mismatches == 0) {
        printf("\nPASS: AXI-ready pooling stage is bit-exact with the plain-array\n");
        printf("reference and the Python F.avg_pool2d reference.\n");
        return 0;
    } else {
        printf("\nFAIL.\n");
        return 1;
    }
}
