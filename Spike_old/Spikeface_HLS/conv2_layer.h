#ifndef CONV2_LAYER_H
#define CONV2_LAYER_H

#include "conv2_types.h"
#include "lif_neuron.h"

#define CONV2_IN_CHANNELS 8
#define CONV2_OUT_CHANNELS 16
#define CONV2_H 16
#define CONV2_W 16
#define CONV2_K 5
#define CONV2_PAD 2

// Parallelism factor: how many of the 8 input channels are processed
// simultaneously per cycle in conv2_layer_step's channel-MAC loop. Must
// evenly divide CONV2_IN_CHANNELS (valid: 1, 2, 4, 8). Higher = more DSP/
// LUT, lower latency -- but also deeper combinational logic per cycle.
//
// CONV2_IC_GROUP=4 was tried and hit NEGATIVE SLACK (-6.98 ns against the
// 10 ns target): 4 parallel 25-tap MACs feeding one adder tree in a single
// cycle is too much combinational depth to close timing at 100 MHz. Rather
// than chase a timing fix on top of that trade-off, reverted to
// CONV2_IC_GROUP=1 -- the verified configuration with POSITIVE slack
// (7.268 ns estimated vs. 10 ns target) and comfortable resource margins
// (DSP=26/220, LUT=9368/53200). Accepting the higher latency (see
// conv2_layer.cpp header comment) in exchange for a design that's
// actually correct and timing-clean, rather than one that "looks fast on
// paper" but wouldn't run reliably on real silicon.
#define CONV2_IC_GROUP 1

// Helper functions (inline, header-only -- same pattern as lif_update /
// event_gated_mac25 in lif_neuron.h). These are internal building blocks,
// not top functions, so 'inline' is correct and safe here.

// Event-gated MAC over one input channel's 5x5 receptive field, generalized
// to the pooled (5-level) activation type.
inline acc_t event_gated_mac25_pooled(const pooled_t patch[25], const weight_t kernel[25], int &active_taps) {
#pragma HLS INLINE
    acc_t acc = 0;
    active_taps = 0;
MAC_LOOP:
    for (int i = 0; i < 25; i++) {
#pragma HLS UNROLL
        if (patch[i] != pooled_t(0)) {
            acc += acc_t(patch[i]) * acc_t(kernel[i]);
            active_taps++;
        }
    }
    return acc;
}

// One conv2 neuron, one timestep: sums event-gated MAC contributions across
// all 8 input channels (200 taps total), then applies the same LIF update
// used everywhere else in the design.
inline spike_t conv2_neuron_step(
    const pooled_t patch[CONV2_IN_CHANNELS][25],
    const weight_t kernel[CONV2_IN_CHANNELS][25],
    mem_t &mem,
    mem_t beta,
    mem_t threshold,
    int &active_taps
) {
#pragma HLS INLINE off
    acc_t total_cur = 0;
    active_taps = 0;
IC_LOOP:
    for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++) {
        int taps_ic = 0;
        acc_t cur = event_gated_mac25_pooled(patch[ic], kernel[ic], taps_ic);
        total_cur += cur;
        active_taps += taps_ic;
    }
    return lif_update(total_cur, mem, beta, threshold);
}

// Top function: declaration only. No 'inline', no logic body -- implemented
// in conv2_layer.cpp.
void conv2_layer_step(
    const pooled_t input_frame[CONV2_IN_CHANNELS][CONV2_H][CONV2_W],
    const weight_t weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][CONV2_K][CONV2_K],
    mem_t mem_state[CONV2_OUT_CHANNELS][CONV2_H][CONV2_W],
    spike_t spike_out[CONV2_OUT_CHANNELS][CONV2_H][CONV2_W],
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
);

#endif
