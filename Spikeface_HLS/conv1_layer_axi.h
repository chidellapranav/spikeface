#ifndef CONV1_LAYER_AXI_H
#define CONV1_LAYER_AXI_H

#include "spikeface_types.h"
#include "lif_neuron.h"
#include "spikeface_neuron_top.h"
#include "hls_stream.h"

#define CONV1_OUT_CHANNELS 8
#define CONV1_H 32
#define CONV1_W 32
#define CONV1_K 5
#define CONV1_PAD 2

// Declaration only. No 'inline', no logic body.
void conv1_layer_axi(
    hls::stream<spike_t> &input_frame,
    const weight_t weights[CONV1_OUT_CHANNELS][CONV1_K][CONV1_K],
    hls::stream<spike_t> &spike_out,
    mem_t beta,
    mem_t threshold,
    long &total_active_taps
);

#endif
