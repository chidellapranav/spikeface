#include "lfw_model.h"

void lfw_conv1_axi(hls::stream<float>& input,const float *params,
                   hls::stream<lfw_spike_t>& output,float beta,float threshold,ap_uint<1> clear_state) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=PARAM depth=224
#pragma HLS INTERFACE s_axilite port=params bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=clear_state bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
 static float membrane[8192];float image[4096];lfw_spike_t spikes[8192];
 for(int i=0;i<4096;i++)image[i]=input.read();
 lfw_conv1_step(image,params,membrane,spikes,beta,threshold,clear_state);
 for(int i=0;i<8192;i++)output.write(spikes[i]);
}

void lfw_conv2_axi(hls::stream<lfw_spike_t>& input,const float *params,
                   hls::stream<lfw_spike_t>& output,float beta,float threshold,ap_uint<1> clear_state) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=PARAM depth=3248
#pragma HLS INTERFACE s_axilite port=params bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=threshold bundle=CTRL
#pragma HLS INTERFACE s_axilite port=clear_state bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
 static float membrane[4096];lfw_spike_t frame[8192],spikes[4096];
 for(int i=0;i<8192;i++)frame[i]=input.read();
 lfw_conv2_step(frame,params,membrane,spikes,beta,threshold,clear_state);
 for(int i=0;i<4096;i++)output.write(spikes[i]);
}

void lfw_readout_axi(hls::stream<lfw_spike_t>& input,const float *params,
                     hls::stream<lfw_packet_t>& output,float beta,ap_uint<1> clear_state) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=PARAM depth=217247
#pragma HLS INTERFACE s_axilite port=params bundle=CTRL
#pragma HLS INTERFACE s_axilite port=beta bundle=CTRL
#pragma HLS INTERFACE s_axilite port=clear_state bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
 static float membrane[53],evidence[53];static unsigned steps=0;
 lfw_spike_t frame[4096];float scores[53];
 for(int i=0;i<4096;i++)frame[i]=input.read();
 lfw_readout_step(frame,params,membrane,evidence,steps,scores,beta,clear_state);
 for(int i=0;i<53;i++) {
  union {float value;unsigned bits;} cast;cast.value=scores[i];
  lfw_packet_t packet;packet.data=cast.bits;packet.keep=-1;packet.strb=-1;packet.last=i==52;
  output.write(packet);
 }
}
