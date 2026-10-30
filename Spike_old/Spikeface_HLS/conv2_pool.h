#ifndef CONV2_POOL_H
#define CONV2_POOL_H

#include "conv2_types.h"

#define POOL2_CHANNELS 16
#define POOL2_IN_H 16
#define POOL2_IN_W 16
#define POOL2_OUT_H 8
#define POOL2_OUT_W 8

// Declaration only. No 'inline', no logic body.
//
// Mirrors conv1_pool_step exactly (2x2 average pooling, no LIF/threshold
// -- purely a spatial downsample), resized for conv2's output: 16 channels
// x 16x16 -> 16 channels x 8x8, feeding fc1.
void conv2_pool_step(
    const spike_t input_frame[POOL2_CHANNELS][POOL2_IN_H][POOL2_IN_W],
    pooled_t output_frame[POOL2_CHANNELS][POOL2_OUT_H][POOL2_OUT_W]
);

#endif
