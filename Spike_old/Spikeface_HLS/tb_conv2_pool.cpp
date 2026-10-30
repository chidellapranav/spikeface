#include <cstdio>
#include "conv2_pool.h"
#include "golden_pool2.h"

int main() {
    static spike_t input_frame[POOL2_CHANNELS][POOL2_IN_H][POOL2_IN_W];
    static pooled_t output_frame[POOL2_CHANNELS][POOL2_OUT_H][POOL2_OUT_W];

    long total_mismatches = 0;
    double max_abs_err = 0.0;

    for (int t = 0; t < GOLDEN_POOL2_T; t++) {
        for (int ch = 0; ch < POOL2_CHANNELS; ch++)
            for (int y = 0; y < POOL2_IN_H; y++)
                for (int x = 0; x < POOL2_IN_W; x++)
                    input_frame[ch][y][x] = (spike_t)GOLDEN_POOL2_INPUT[t][ch][y][x];

        conv2_pool_step(input_frame, output_frame);

        for (int ch = 0; ch < POOL2_CHANNELS; ch++) {
            for (int y = 0; y < POOL2_OUT_H; y++) {
                for (int x = 0; x < POOL2_OUT_W; x++) {
                    double got = (double)output_frame[ch][y][x];
                    double expected = GOLDEN_POOL2_OUTPUT[t][ch][y][x];
                    double err = got - expected;
                    if (err < 0) err = -err;
                    if (err > max_abs_err) max_abs_err = err;
                    if (err > 1e-9) {
                        total_mismatches++;
                        if (total_mismatches <= 10)
                            printf("MISMATCH t=%d ch=%d y=%d x=%d got=%f expected=%f\n",
                                   t, ch, y, x, got, expected);
                    }
                }
            }
        }
    }

    printf("Total values checked: %d x %d = %ld\n",
           GOLDEN_POOL2_T, POOL2_CHANNELS*POOL2_OUT_H*POOL2_OUT_W,
           (long)GOLDEN_POOL2_T*POOL2_CHANNELS*POOL2_OUT_H*POOL2_OUT_W);
    printf("Mismatches: %ld\n", total_mismatches);
    printf("Max abs error: %.10f\n", max_abs_err);

    if (total_mismatches == 0) {
        printf("\nPASS: conv2 pooling stage is bit-exact with the Python F.avg_pool2d reference.\n");
        return 0;
    } else {
        printf("\nFAIL.\n");
        return 1;
    }
}
