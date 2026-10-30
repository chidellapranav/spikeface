#ifndef CONV2_POOL_AXI_H
#define CONV2_POOL_AXI_H

#include "conv2_types.h"
#include "hls_stream.h"

#define POOL2_CHANNELS 16
#define POOL2_IN_H 16
#define POOL2_IN_W 16
#define POOL2_OUT_H 8
#define POOL2_OUT_W 8

// Declaration only. No 'inline', no logic body -- required for any top
// function using hls::stream<T>.
//
// input_frame: 16*16*16 = 4096 spikes, channel-major then raster order --
//   matches exactly how conv2_layer_axi's spike_out stream is written.
// output_frame: 16*8*8 = 1024 pooled values, same ordering -- this is
//   what fc1's AXI-ready version will expect to read.
void conv2_pool_axi(
    hls::stream<spike_t> &input_frame,
    hls::stream<pooled_t> &output_frame
);

#endif
