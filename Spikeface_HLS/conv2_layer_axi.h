#ifndef CONV2_LAYER_AXI_H
#define CONV2_LAYER_AXI_H

#include "conv2_layer.h"
#include "hls_stream.h"

// Declaration only. No 'inline', no logic body -- required for any top
// function using hls::stream<T> (see conv1_layer_axi.h for why).
//
// input_frame: 8*16*16 = 2048 pooled values, channel-major then raster
//   order -- matches exactly how conv1_pool_axi's output_frame stream is
//   written.
// spike_out: 16*16*16 = 4096 spikes, same channel-major/raster ordering --
//   this is what the (future) conv2_pool_axi will expect to read.
void conv2_layer_axi(
    hls::stream<pooled_t> &input_frame,
    const weight_t weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][CONV2_K][CONV2_K],
    hls::stream<spike_t> &spike_out,
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
);

#endif
