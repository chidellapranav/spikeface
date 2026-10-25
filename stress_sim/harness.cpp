// Binary stdin/stdout bridge to the actual HLS, using unmodified Xilinx types.
// No arithmetic reference is implemented here. Integers on the wire are raw codes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "conv1_layer.h"
#include "conv1_layer_axi.h"
#include "conv1_pool.h"
#include "conv1_pool_axi.h"
#include "conv2_layer_axi.h"
#include "conv2_pool.h"
#include "conv2_pool_axi.h"
#include "fc1_layer_axi.h"
#include "fc2_layer_axi.h"

int32_t get() { int32_t x; if (fread(&x, 4, 1, stdin)!=1) throw std::runtime_error("short input"); return x; }
void put(int64_t v) { int32_t x=v; if(fwrite(&x,4,1,stdout)!=1) throw std::runtime_error("write failed"); }
template<class T> T code(int32_t x) { T v; v.range(v.length()-1,0)=x; return v; }
template<class T> int32_t raw(T x) { return x.range(x.length()-1,0).to_int(); }
template<class T> void load(T *p, int n) { for(int i=0;i<n;i++) p[i]=code<T>(get()); }
template<class T> void dump(T *p, int n) { for(int i=0;i<n;i++) put(raw(p[i])); }
template<class T> void feed(hls::stream<T>& s,T *p,int n) { for(int i=0;i<n;i++) s.write(p[i]); }
template<class T> void drain(hls::stream<T>& s,int n) { for(int i=0;i<n;i++) put(raw(s.read())); if(!s.empty()) throw std::runtime_error("excess output"); }
template<class T> void capture(hls::stream<T>& s,T *p,int n) { for(int i=0;i<n;i++) {p[i]=s.read();put(raw(p[i]));} if(!s.empty()) throw std::runtime_error("excess output"); }
void scores(hls::stream<score_packet_t>& s) {
  for(int i=0;i<40;i++) {auto p=s.read(); put(p.data.to_int()); put(p.last.to_int()); put(p.keep.to_int()); put(p.strb.to_int());}
  if(!s.empty()) throw std::runtime_error("excess scores");
}
static weight_t w1[8][5][5],w2[16][8][5][5],wf[64][1024];
static weight_fc2_t wr[40][64];
static mem_t m1[8][32][32],m2[16][16][16],mf[64];
static spike_t x1[32][32],c1[8][32][32],c2[16][16][16],f[64];
static pooled_t p1[8][16][16],p2[16][8][8];
static spike_t ac1[8192],ac2[4096],af[64];
static pooled_t ap1[2048],ap2[1024];

int main(int argc,char**argv) { try {
 if(argc!=2) return 2; std::string mode=argv[1]; int n=get();
 if(mode=="lif") {for(int i=0;i<n;i++) { auto cur=code<acc_t>(get()); auto m=code<mem_t>(get()); auto b=code<mem_t>(get()); auto t=code<mem_t>(get()); put(raw(lif_update(cur,m,b,t)));put(raw(m));}return 0;}
 auto beta=code<mem_t>(get()),thr=code<mem_t>(get());
 if(mode=="conv1"||mode=="chain")load(&w1[0][0][0],200);
 if(mode=="conv2"||mode=="chain")load(&w2[0][0][0][0],3200);
 if(mode=="fc1"||mode=="chain")load(&wf[0][0],65536);
 if(mode=="fc2"||mode=="chain")load(&wr[0][0],2560);
 for(int t=0;t<n;t++) {
  int clear_state=get();
  if(clear_state) {for(auto &ch:m1)for(auto &row:ch)for(auto &v:row)v=0;
    for(auto &ch:m2)for(auto &row:ch)for(auto &v:row)v=0;for(auto &v:mf)v=0;}
  long taps=0,at=0;
  if(mode=="conv1"||mode=="chain") {
   load(&x1[0][0],1024);conv1_layer_step(x1,w1,m1,c1,beta,thr,taps);
   hls::stream<spike_t> in,out;feed(in,&x1[0][0],1024);conv1_layer_axi(in,w1,out,beta,thr,at,clear_state);
   dump(&c1[0][0][0],8192);capture(out,ac1,8192);dump(&m1[0][0][0],8192);put(taps);put(at);
  }
  if(mode=="pool1"||mode=="chain") {
   if(mode=="pool1")load(&c1[0][0][0],8192);conv1_pool_step(c1,p1);
   hls::stream<spike_t> in;hls::stream<pooled_t> out;feed(in,mode=="chain"?ac1:&c1[0][0][0],8192);conv1_pool_axi(in,out);
   dump(&p1[0][0][0],2048);capture(out,ap1,2048);
  }
  if(mode=="conv2"||mode=="chain") {
   if(mode=="conv2")load(&p1[0][0][0],2048);conv2_layer_step(p1,w2,m2,c2,beta,thr,taps);
   hls::stream<pooled_t> in;hls::stream<spike_t> out;feed(in,mode=="chain"?ap1:&p1[0][0][0],2048);conv2_layer_axi(in,w2,out,beta,thr,at,clear_state);
   dump(&c2[0][0][0],4096);capture(out,ac2,4096);dump(&m2[0][0][0],4096);put(taps);put(at);
  }
  if(mode=="pool2"||mode=="chain") {
   if(mode=="pool2")load(&c2[0][0][0],4096);conv2_pool_step(c2,p2);
   hls::stream<spike_t> in;hls::stream<pooled_t> out;feed(in,mode=="chain"?ac2:&c2[0][0][0],4096);conv2_pool_axi(in,out);
   dump(&p2[0][0][0],1024);capture(out,ap2,1024);
  }
  if(mode=="fc1"||mode=="chain") {
   if(mode=="fc1")load(&p2[0][0][0],1024);fc1_layer_step(&p2[0][0][0],wf,mf,f,beta,thr,taps);
   hls::stream<pooled_t> in;hls::stream<spike_t> out;feed(in,mode=="chain"?ap2:&p2[0][0][0],1024);fc1_layer_axi(in,wf,out,beta,thr,at,clear_state);
   dump(&f[0],64);capture(out,af,64);dump(&mf[0],64);put(taps);put(at);
  }
  if(mode=="fc2"||mode=="chain") {
   if(mode=="fc2")load(f,64);int clear=get();
   hls::stream<spike_t> in;hls::stream<score_packet_t> out;feed(in,mode=="chain"?af:f,64);fc2_layer_axi(in,wr,clear,out);scores(out);
  }
 }
 if(fgetc(stdin)!=EOF)throw std::runtime_error("trailing input");
 return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 3;}}
