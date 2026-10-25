// Verification-only composite of the three production HLS tops. The small
// FIFOs exist in generated RTL, so this tests real pipeline backpressure.
#include "lfw_model.h"
void lfw_chain_verify(hls::stream<float>& input,const float *params1,
 const float *params2,const float *params3,hls::stream<lfw_packet_t>& output,
 float beta,float threshold,ap_uint<1> clear_state) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE m_axi port=params1 offset=slave bundle=PARAM1 depth=224
#pragma HLS INTERFACE m_axi port=params2 offset=slave bundle=PARAM2 depth=3248
#pragma HLS INTERFACE m_axi port=params3 offset=slave bundle=PARAM3 depth=217247
#pragma HLS INTERFACE s_axilite port=params1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=params2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=params3 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=clear_state bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW
 hls::stream<lfw_spike_t> conv1_spikes("conv1_spikes"),conv2_spikes("conv2_spikes");
#pragma HLS STREAM variable=conv1_spikes depth=16
#pragma HLS STREAM variable=conv2_spikes depth=16
 lfw_conv1_axi(input,params1,conv1_spikes,beta,threshold,clear_state);
 lfw_conv2_axi(conv1_spikes,params2,conv2_spikes,beta,threshold,clear_state);
 lfw_readout_axi(conv2_spikes,params3,output,beta,clear_state);
}
