// Native RTL campaign: two timestep images, repeated with per-image clear,
// and XSIM's random interface stalls. All expected outputs are fixture data.
#include "lfw_model.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
static float p1[224],p2[3248],p3[217247],image[4096];
struct Expected {uint32_t s1[8192],s2[4096];float scores[53];};
template<class T>bool load(FILE*f,T*p,int n){return fread(p,sizeof(T),n,f)==unsigned(n);}
int main(){
 FILE*f=fopen("lfw_fixture.bin","rb");if(!f){printf("Missing fixture\n");return 2;}
 unsigned steps=0;
 if(!load(f,&steps,1)||steps<2||steps>25||!load(f,p1,224)||!load(f,p2,3248)||!load(f,p3,217247)||!load(f,image,4096)){fclose(f);return 2;}
 std::vector<Expected> golden(steps);
 for(unsigned t=0;t<steps;t++)if(!load(f,golden[t].s1,8192)||!load(f,golden[t].s2,4096)||!load(f,golden[t].scores,53)){fclose(f);return 2;}
 fclose(f);unsigned errors=0;
 for(unsigned image_index=0;image_index<2;image_index++)for(unsigned t=0;t<steps;t++){
  hls::stream<float>in;hls::stream<lfw_spike_t>a,b,c,d;hls::stream<lfw_packet_t>out;
  for(int i=0;i<4096;i++)in.write(image[i]);
  lfw_conv1_axi(in,p1,a,.9375f,.5f,t==0);
  for(int i=0;i<8192;i++){auto v=a.read();if(v!=golden[t].s1[i])errors++;b.write(v);}
  lfw_conv2_axi(b,p2,c,.9375f,.5f,t==0);
  for(int i=0;i<4096;i++){auto v=c.read();if(v!=golden[t].s2[i])errors++;d.write(v);}
  lfw_readout_axi(d,p3,out,.9375f,t==0);
  for(int i=0;i<53;i++){
   auto packet=out.read();uint32_t bits=0;memcpy(&bits,&golden[t].scores[i],4);
   // Scores are compared bit-for-bit, including all packet sidebands.
   if(packet.data.to_uint()!=bits||packet.last!=(i==52)||packet.keep!=15||packet.strb!=15)errors++;
  }
  if(!in.empty()||!a.empty()||!b.empty()||!c.empty()||!d.empty()||!out.empty())errors++;
 }
 printf("%s: RTL stress fixture, %u transactions, %u mismatches\n",errors?"FAIL":"PASS",2*steps,errors);
 return errors?1:0;
}
