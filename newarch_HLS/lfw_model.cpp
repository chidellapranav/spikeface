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
   // Interleave the four pooling positions so each floating-point sum
   // has four cycles between updates, preserving its original tap order.
   float sums[4]={0.0f,0.0f,0.0f,0.0f};
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
   conv1_taps: for(int i=0;i<100;i++) {
#pragma HLS PIPELINE II=1
     int p=i%4,tap=i/4,ky=tap/5,kx=tap%5;
     int dy=p/2,dx=p%2;
     int iy=oy*2+dy+ky-2,ix=ox*2+dx+kx-2;
     float value=(iy>=0&&iy<64&&ix>=0&&ix<64)?image[iy*64+ix]:0.0f;
     // Rotate the registers explicitly: the updated sum returns to the
     // adder after four iterations. This also exposes the true dependency
     // to HLS without suppressing any dependence checks.
     float next=sums[0]+value*kernel[ky*5+kx];
     sums[0]=sums[1];sums[1]=sums[2];sums[2]=sums[3];sums[3]=next;
   }
   for(int p=0;p<4;p++) {
    float sum=sums[p];
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
   // Four independent pooling positions, with the original tap order.
   float sums[4]={0.0f,0.0f,0.0f,0.0f};
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
   conv2_taps: for(int i=0;i<800;i++) {
#pragma HLS PIPELINE II=1
     int p=i%4,tap=i/4,ic=tap/25,ky=(tap%25)/5,kx=tap%5;
     int dy=p/2,dx=p%2;
     int iy=oy*2+dy+ky-2,ix=ox*2+dx+kx-2;
     bool active=(iy>=0&&iy<32&&ix>=0&&ix<32)&&input[ic*1024+iy*32+ix]!=0;
     float next=sums[0]+(active?kernel[tap]:0.0f);
     sums[0]=sums[1];sums[1]=sums[2];sums[2]=sums[3];sums[3]=next;
   }
   for(int p=0;p<4;p++) {
    float sum=sums[p];
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
 // Separate contiguous AXI reads avoid three requests per output iteration.
 float biases[53],scales[53],shifts[53];
 readout_biases: for(int oc=0;oc<53;oc++) {
#pragma HLS PIPELINE II=1
  biases[oc]=params[217088+oc];
 }
 readout_scales: for(int oc=0;oc<53;oc++) {
#pragma HLS PIPELINE II=1
  scales[oc]=params[217141+oc];
 }
 readout_shifts: for(int oc=0;oc<53;oc++) {
#pragma HLS PIPELINE II=1
  shifts[oc]=params[217194+oc];
 }
 // Interleave independent output classes rather than reassociating a sum.
 // The last group has one valid class; padded lanes never access params.
 for(int base=0;base<53;base+=4) {
  float sums[4]={0.0f,0.0f,0.0f,0.0f};
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
  readout_taps: for(int j=0;j<4096*4;j++) {
#pragma HLS PIPELINE II=1
   int p=j%4,i=j/4,oc=base+p;
   float next=sums[0]+((oc<53&&input[i]!=0)?params[oc*4096+i]:0.0f);
   sums[0]=sums[1];sums[1]=sums[2];sums[2]=sums[3];sums[3]=next;
  }
  for(int p=0;p<4;p++) {
   int oc=base+p;
   if(oc>=53)continue;
   float sum=sums[p];
  float cur=(sum+biases[oc])*scales[oc]+shifts[oc];
  float updated=beta*(clear_state?0.0f:membrane[oc])+cur;
  membrane[oc]=updated;
  evidence[oc]=(clear_state?0.0f:evidence[oc])+updated;
  scores[oc]=evidence[oc]/float(steps);
  }
 }
}
