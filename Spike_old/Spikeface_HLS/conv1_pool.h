#ifndef CONV1_POOL_H
#define CONV1_POOL_H

#include "conv2_types.h"

#define POOL_CHANNELS 8
#define POOL_IN_H 32
#define POOL_IN_W 32
#define POOL_OUT_H 16
#define POOL_OUT_W 16

// Declaration only. No 'inline', no logic body.
void conv1_pool_step(
    const spike_t input_frame[POOL_CHANNELS][POOL_IN_H][POOL_IN_W],
    pooled_t output_frame[POOL_CHANNELS][POOL_OUT_H][POOL_OUT_W]
);

#endif
