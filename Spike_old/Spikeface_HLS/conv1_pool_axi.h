#ifndef CONV1_POOL_AXI_H
#define CONV1_POOL_AXI_H

#include "conv2_types.h"
#include "hls_stream.h"

#define POOL_CHANNELS 8
#define POOL_IN_H 32
#define POOL_IN_W 32
#define POOL_OUT_H 16
#define POOL_OUT_W 16

// Declaration only. No 'inline', no logic body -- required for any top
// function using hls::stream<T> (see conv1_layer_axi.h for why).
//
// input_frame: 8*32*32 = 2048 spikes, channel-major then raster order --
//   matches exactly how conv1_layer_axi's spike_out stream is written.
// output_frame: 8*16*16 = 512 pooled values, same channel-major/raster
//   ordering -- this is what conv2_layer_axi will expect to read.
void conv1_pool_axi(
    hls::stream<spike_t> &input_frame,
    hls::stream<pooled_t> &output_frame
);

#endif
