#include "lfw_model.h"
#include <cstdint>
#include <cstdio>
#include <cmath>
static float p1[224],p2[3248],p3[217247],image[4096];
static uint32_t s1[8192],s2[4096];static float expected[53];
template<class T>bool load(FILE*f,T*p,int n){return fread(p,sizeof(T),n,f)==unsigned(n);}
int main(){
 FILE*f=fopen("lfw_fixture.bin","rb");if(!f){printf("Missing lfw_fixture.bin\n");return 2;}
 unsigned steps=0;if(!load(f,&steps,1)||steps<1||steps>25||!load(f,p1,224)||!load(f,p2,3248)||!load(f,p3,217247)||!load(f,image,4096)){fclose(f);return 2;}
 unsigned errors=0;
 for(unsigned t=0;t<steps;t++){
  if(!load(f,s1,8192)||!load(f,s2,4096)||!load(f,expected,53)){fclose(f);return 2;}
  hls::stream<float> in;hls::stream<lfw_spike_t>a,b,c,d;hls::stream<lfw_packet_t>out;
  for(int i=0;i<4096;i++)in.write(image[i]);
  lfw_conv1_axi(in,p1,a,.9375f,.5f,t==0);
  for(int i=0;i<8192;i++){auto value=a.read();if(value!=s1[i])errors++;b.write(value);}
  lfw_conv2_axi(b,p2,c,.9375f,.5f,t==0);
  for(int i=0;i<4096;i++){auto value=c.read();if(value!=s2[i])errors++;d.write(value);}
  lfw_readout_axi(d,p3,out,.9375f,t==0);
  for(int i=0;i<53;i++){
   auto packet=out.read();union{uint32_t bits;float value;}actual;actual.bits=packet.data.to_uint();
   if(!std::isfinite(actual.value)||std::fabs(actual.value-expected[i])>1e-4f+1e-4f*std::fabs(expected[i]))errors++;
   if(packet.last!=(i==52)||packet.keep!=15||packet.strb!=15)errors++;
  }
 }
 fclose(f);printf("%s: LFW chain, %u timesteps, %u mismatches\n",errors?"FAIL":"PASS",steps,errors);
 return errors?1:0;
}
