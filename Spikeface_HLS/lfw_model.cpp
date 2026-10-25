#include "lfw_model.h"
#include <cfloat>

static float lif_subtract(float current,float previous,float beta,float threshold) {
#pragma HLS INLINE
    // snnTorch Leaky's default is delayed subtractive reset, strict > threshold.
    // The reset depends on PREVIOUS membrane, and no Q3.5 clipping occurs.
    float integrated=beta*previous+current;
    return integrated-((previous>threshold)?threshold:0.0f);
}

void lfw_conv1_step(const float image[4096],const float params[LFW_C1_PARAMS],
                    float membrane[8192],lfw_spike_t spikes[8192],float beta,
                    float threshold,bool clear_state) {
#pragma HLS INLINE off
 for(int oc=0;oc<8;oc++) {
  float kernel[25];for(int k=0;k<25;k++)kernel[k]=params[oc*25+k];
  float bias=params[200+oc],scale=params[208+oc],shift=params[216+oc];
  for(int oy=0;oy<32;oy++)for(int ox=0;ox<32;ox++) {
   float best=-FLT_MAX;
   for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++) {
    float sum=0;
    for(int ky=0;ky<5;ky++)for(int kx=0;kx<5;kx++) {
#pragma HLS PIPELINE II=1
     int iy=oy*2+dy+ky-2,ix=ox*2+dx+kx-2;
     float value=(iy>=0&&iy<64&&ix>=0&&ix<64)?image[iy*64+ix]:0.0f;
     sum+=value*kernel[ky*5+kx];
    }
    float cur=(sum+bias)*scale+shift;
    if(cur>best)best=cur;
   }
   int idx=oc*1024+oy*32+ox;
   float updated=lif_subtract(best,clear_state?0.0f:membrane[idx],beta,threshold);
   membrane[idx]=updated;spikes[idx]=updated>threshold;
  }
 }
}

void lfw_conv2_step(const lfw_spike_t input[8192],const float params[LFW_C2_PARAMS],
                    float membrane[4096],lfw_spike_t spikes[4096],float beta,
                    float threshold,bool clear_state) {
#pragma HLS INLINE off
 for(int oc=0;oc<16;oc++) {
  float kernel[200];for(int k=0;k<200;k++)kernel[k]=params[oc*200+k];
  float bias=params[3200+oc],scale=params[3216+oc],shift=params[3232+oc];
  for(int oy=0;oy<16;oy++)for(int ox=0;ox<16;ox++) {
   float best=-FLT_MAX;
   for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++) {
    float sum=0;
    for(int ic=0;ic<8;ic++)for(int ky=0;ky<5;ky++)for(int kx=0;kx<5;kx++) {
#pragma HLS PIPELINE II=1
     int iy=oy*2+dy+ky-2,ix=ox*2+dx+kx-2;
     bool active=(iy>=0&&iy<32&&ix>=0&&ix<32)&&input[ic*1024+iy*32+ix]!=0;
     sum+=active?kernel[ic*25+ky*5+kx]:0.0f;
    }
    float cur=(sum+bias)*scale+shift;
    if(cur>best)best=cur;
   }
   int idx=oc*256+oy*16+ox;
   float updated=lif_subtract(best,clear_state?0.0f:membrane[idx],beta,threshold);
   membrane[idx]=updated;spikes[idx]=updated>threshold;
  }
 }
}

void lfw_readout_step(const lfw_spike_t input[4096],const float params[LFW_FC_PARAMS],
                     float membrane[53],float evidence[53],unsigned &steps,
                     float scores[53],float beta,bool clear_state) {
#pragma HLS INLINE off
 if(clear_state)steps=0;
 steps++;
 for(int oc=0;oc<53;oc++) {
  float sum=0;
  for(int i=0;i<4096;i++) {
#pragma HLS PIPELINE II=1
   sum+=input[i]!=0?params[oc*4096+i]:0.0f;
  }
  float cur=(sum+params[217088+oc])*params[217141+oc]+params[217194+oc];
  float updated=beta*(clear_state?0.0f:membrane[oc])+cur;
  membrane[oc]=updated;
  evidence[oc]=(clear_state?0.0f:evidence[oc])+updated;
  scores[oc]=evidence[oc]/float(steps);
 }
}
