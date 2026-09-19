#ifndef FC1_LAYER_AXI_H
#define FC1_LAYER_AXI_H

#include "fc1_layer.h"
#include "hls_stream.h"

// Declaration only. No 'inline', no logic body -- required for any top
// function using hls::stream<T>.
//
// input_flat: 1024 pooled values, in the same flat channel-major/row-major
//   order as fc1_layer_step -- matches conv2_pool_axi's output exactly.
// spike_out: 64 spikes.
void fc1_layer_axi(
    hls::stream<pooled_t> &input_flat,
    const weight_t weights[FC1_OUT][FC1_IN],
    hls::stream<spike_t> &spike_out,
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
);

#endif
