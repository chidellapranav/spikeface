#ifndef SPIKEFACE_LFW_MODEL_H
#define SPIKEFACE_LFW_MODEL_H
// Authoritative LFW architecture. Float32 intentionally preserves the supplied
// model's arithmetic domain; this is not the legacy 8-bit ORL accelerator.
#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
typedef ap_uint<8> lfw_spike_t;
typedef ap_axiu<32,0,0,0> lfw_packet_t;
const int LFW_CLASSES=53;
const int LFW_C1_PARAMS=200+3*8;
const int LFW_C2_PARAMS=3200+3*16;
const int LFW_FC_PARAMS=53*4096+3*53;

// Parameters: flattened weights, convolution/linear bias, BN scale, BN shift.
// BN affine is applied AFTER the original convolution, before max pooling.
void lfw_conv1_step(const float image[4096], const float params[LFW_C1_PARAMS],
                    float membrane[8192], lfw_spike_t spikes[8192], float beta,
                    float threshold, bool clear_state);
void lfw_conv2_step(const lfw_spike_t input[8192], const float params[LFW_C2_PARAMS],
                    float membrane[4096], lfw_spike_t spikes[4096], float beta,
                    float threshold, bool clear_state);
void lfw_readout_step(const lfw_spike_t input[4096], const float params[LFW_FC_PARAMS],
                     float membrane[53], float evidence[53], unsigned &steps,
                     float scores[53], float beta, bool clear_state);

void lfw_conv1_axi(hls::stream<float>& input, const float *params,
                   hls::stream<lfw_spike_t>& output, float beta, float threshold,
                   ap_uint<1> clear_state);
void lfw_conv2_axi(hls::stream<lfw_spike_t>& input, const float *params,
                   hls::stream<lfw_spike_t>& output, float beta, float threshold,
                   ap_uint<1> clear_state);
void lfw_readout_axi(hls::stream<lfw_spike_t>& input, const float *params,
                     hls::stream<lfw_packet_t>& output, float beta,
                     ap_uint<1> clear_state);
#endif
