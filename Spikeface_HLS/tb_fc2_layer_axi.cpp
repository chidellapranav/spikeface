#include <cstdio>
#include <cmath>
#include "fc2_layer_axi.h"
#include "golden_fc2.h"

int main() {
    static weight_fc2_t weights[FC2_OUT][FC2_IN];
    for (int o = 0; o < FC2_OUT; o++)
        for (int i = 0; i < FC2_IN; i++)
            weights[o][i] = (weight_fc2_t)GOLDEN_FC2_WEIGHTS[o][i];

    long total_mismatches = 0;
    double max_abs_err = 0.0;

    printf("%-4s | %-12s | %-14s | %-10s\n", "t", "argmax(HLS)", "argmax(golden)", "match?");
    printf("--------------------------------------------------\n");

    for (int t = 0; t < GOLDEN_FC2_T; t++) {
        hls::stream<spike_t> fc1_spikes_in;
        hls::stream<score_packet_t> class_scores_out;

        for (int i = 0; i < FC2_IN; i++)
            fc1_spikes_in.write((spike_t)GOLDEN_FC2_INPUT[t][i]);

        ap_uint<1> clear_accum = (t == 0) ? 1 : 0;  // reset only on first timestep
        fc2_layer_axi(fc1_spikes_in, weights, clear_accum, class_scores_out);

        double hls_scores[FC2_OUT];
        int hls_argmax = 0;
        int last_count = 0;
        for (int o = 0; o < FC2_OUT; o++) {
            score_packet_t pkt = class_scores_out.read();
            // Unpack: reinterpret the raw 32-bit pattern back into the
            // fixed-point score type (bit-pattern copy, not numeric cast).
            axis_score_t score;
            score.range(31, 0) = pkt.data;
            hls_scores[o] = (double)score;
            if (hls_scores[o] > hls_scores[hls_argmax]) hls_argmax = o;

            // TLAST must be asserted on exactly the 40th (last) score,
            // and on no other.
            bool expected_last = (o == FC2_OUT - 1);
            if ((bool)pkt.last != expected_last) {
                printf("TLAST MISMATCH at t=%d o=%d: got=%d expected=%d\n", t, o, (int)pkt.last, expected_last);
                total_mismatches++;
            }
            if (pkt.last) last_count++;

            double expected = GOLDEN_FC2_SCORES[t][o];
            double err = hls_scores[o] - expected;
            if (err < 0) err = -err;
            if (err > max_abs_err) max_abs_err = err;
            if (err > 1e-6) total_mismatches++;
        }
        if (last_count != 1) {
            printf("TLAST COUNT ERROR at t=%d: exactly 1 assertion expected, got %d\n", t, last_count);
            total_mismatches++;
        }

        int golden_argmax = 0;
        for (int o = 1; o < FC2_OUT; o++)
            if (GOLDEN_FC2_SCORES[t][o] > GOLDEN_FC2_SCORES[t][golden_argmax]) golden_argmax = o;

        printf("%-4d | %-12d | %-14d | %-10s\n",
               t, hls_argmax, golden_argmax, (hls_argmax == golden_argmax) ? "OK" : "MISMATCH");
    }

    printf("\n--------------------------------------------------\n");
    printf("Total values checked: %d x %d = %d\n", GOLDEN_FC2_T, FC2_OUT, GOLDEN_FC2_T*FC2_OUT);
    printf("Value mismatches (tol 1e-6): %ld\n", total_mismatches);
    printf("Max abs error: %.10f\n", max_abs_err);

    if (total_mismatches == 0) {
        printf("\nPASS: fc2 readout is bit-exact with the Python fixed-point reference.\n");
        return 0;
    } else {
        printf("\nFAIL: %ld mismatches.\n", total_mismatches);
        return 1;
    }
}
