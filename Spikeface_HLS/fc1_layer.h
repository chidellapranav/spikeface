#ifndef FC1_LAYER_H
#define FC1_LAYER_H

#include "conv2_types.h"
#include "lif_neuron.h"

#define FC1_IN 1024   // 16 channels x 8x8, flattened channel-major/row-major
                      // (matches conv2_pool_axi's output stream order exactly
                      // -- no reordering needed between pool2 and fc1)
#define FC1_OUT 64

// Parallelism factor: how many of the 1024 inputs are processed
// simultaneously per cycle. Must evenly divide FC1_IN. Same lesson as
// conv2_layer: full parallelism (1024) would need ~1024 real multipliers
// per neuron per cycle -- way past budget. Defaulting to 1 (safe,
// time-multiplexed, matches conv2's verified working configuration)
// rather than discovering the same LUT/DSP blowup again.
#define FC1_GROUP 1

// Declaration only. No 'inline', no logic body -- implemented in
// fc1_layer.cpp.
void fc1_layer_step(
    const pooled_t input_flat[FC1_IN],
    const weight_t weights[FC1_OUT][FC1_IN],
    mem_t mem_state[FC1_OUT],
    spike_t spike_out[FC1_OUT],
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
);

#endif
