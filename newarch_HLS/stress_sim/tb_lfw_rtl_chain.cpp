#include "lfw_model.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
void lfw_chain_verify(hls::stream<float>&,const float*,const float*,const float*,
 hls::stream<lfw_packet_t>&,float,float,ap_uint<1>);
static float p1[224],p2[3248],p3[217247],image[4096];
template<class T>bool load(FILE*f,T*p,int n){return fread(p,sizeof(T),n,f)==unsigned(n);}
int main(){
 FILE*f=fopen("lfw_fixture.bin","rb");if(!f)return 2;
 unsigned steps=0;
 if(!load(f,&steps,1)||steps<2||steps>25||!load(f,p1,224)||!load(f,p2,3248)||!load(f,p3,217247)||!load(f,image,4096)){fclose(f);return 2;}
 std::vector<std::array<float,53>>expected(steps);
 for(unsigned t=0;t<steps;t++){
  if(fseek(f,(8192+4096)*4,SEEK_CUR)||!load(f,expected[t].data(),53)){fclose(f);return 2;}
 }
 fclose(f);unsigned errors=0;
 for(unsigned round=0;round<2;round++)for(unsigned t=0;t<steps;t++){
  hls::stream<float>in;hls::stream<lfw_packet_t>out;
  for(int i=0;i<4096;i++)in.write(image[i]);
  lfw_chain_verify(in,p1,p2,p3,out,.9375f,.5f,t==0);
  for(int i=0;i<53;i++){
   auto packet=out.read();uint32_t bits=0;memcpy(&bits,&expected[t][i],4);
   if(packet.data.to_uint()!=bits||packet.last!=(i==52)||packet.keep!=15||packet.strb!=15)errors++;
  }
  if(!in.empty()||!out.empty())errors++;
 }
 printf("%s: integrated RTL chain, %u transactions, %u mismatches\n",errors?"FAIL":"PASS",steps*2,errors);
 return errors?1:0;
}
