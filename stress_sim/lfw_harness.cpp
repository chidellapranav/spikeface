#include "lfw_model.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
static float p1[LFW_C1_PARAMS],p2[LFW_C2_PARAMS],p3[LFW_FC_PARAMS];
static float m1[8192],m2[4096],m3[53],e3[53],image[4096],score[53];
static lfw_spike_t c1[8192],c2[4096],ac1[8192],ac2[4096];
uint32_t read32(){uint32_t v;if(fread(&v,4,1,stdin)!=1)throw std::runtime_error("short input");return v;}
void floats(float*p,int n){if(fread(p,4,n,stdin)!=unsigned(n))throw std::runtime_error("short floats");}
void emit(float*p,int n){fwrite(p,4,n,stdout);}
void integer(uint32_t n){fwrite(&n,4,1,stdout);}
void spikes(lfw_spike_t*p,int n){for(int i=0;i<n;i++)integer(p[i].to_uint());}
template<class T>void feed(hls::stream<T>&s,T*p,int n){for(int i=0;i<n;i++)s.write(p[i]);}
void capture(hls::stream<lfw_spike_t>&s,lfw_spike_t*p,int n){for(int i=0;i<n;i++)p[i]=s.read();spikes(p,n);if(!s.empty())throw std::runtime_error("extra output");}
int main(int argc,char**argv){try{
 if(argc!=2)return 2;std::string mode=argv[1];unsigned n=read32(),steps=0;
 float cfg[5];floats(cfg,5);
 if(mode=="chain"||mode=="conv1")floats(p1,LFW_C1_PARAMS);
 if(mode=="chain"||mode=="conv2")floats(p2,LFW_C2_PARAMS);
 if(mode=="chain"||mode=="readout")floats(p3,LFW_FC_PARAMS);
 for(unsigned t=0;t<n;t++){
  bool clear=read32()!=0;
  if(mode=="chain"||mode=="conv1"){
   floats(image,4096);lfw_conv1_step(image,p1,m1,c1,cfg[0],cfg[1],clear);
   hls::stream<float>in;hls::stream<lfw_spike_t>out;feed(in,image,4096);lfw_conv1_axi(in,p1,out,cfg[0],cfg[1],clear);
   spikes(c1,8192);capture(out,ac1,8192);emit(m1,8192);
  }
  if(mode=="chain"||mode=="conv2"){
   if(mode=="conv2")for(auto &v:c1)v=read32();
   lfw_conv2_step(c1,p2,m2,c2,cfg[2],cfg[3],clear);
   hls::stream<lfw_spike_t>in,out;feed(in,mode=="chain"?ac1:c1,8192);lfw_conv2_axi(in,p2,out,cfg[2],cfg[3],clear);
   spikes(c2,4096);capture(out,ac2,4096);emit(m2,4096);
  }
  if(mode=="chain"||mode=="readout"){
   if(mode=="readout")for(auto &v:c2)v=read32();
   lfw_readout_step(c2,p3,m3,e3,steps,score,cfg[4],clear);
   hls::stream<lfw_spike_t>in;hls::stream<lfw_packet_t>out;feed(in,mode=="chain"?ac2:c2,4096);lfw_readout_axi(in,p3,out,cfg[4],clear);
   emit(score,53);emit(m3,53);emit(e3,53);
   for(int i=0;i<53;i++){auto packet=out.read();integer(packet.data.to_uint());integer(packet.last.to_uint());integer(packet.keep.to_uint());integer(packet.strb.to_uint());}
   if(!out.empty())throw std::runtime_error("extra score output");
  }
 }
 if(fgetc(stdin)!=EOF)throw std::runtime_error("trailing input");return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
