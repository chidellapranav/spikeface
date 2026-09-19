#ifndef FC2_LAYER_AXI_H
#define FC2_LAYER_AXI_H

#include "conv2_types.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define FC2_IN 64
#define FC2_OUT 40

// fc2's quantized weights need 2 integer bits, NOT 1 like every other
// layer's weight_t (ap_fixed<8,1>). Found via a real bug: the shared
// normalization chain (see quantization_sweep.py's methodology) computed
// fc2's actual required int_bits=2 from the start -- printed explicitly
// during golden-vector export -- but weight_t was reused here without
// checking that held for fc2 too. Casting fc2's real weight values (up to
// 1.484375 in magnitude) into weight_t's [-1, 0.9921875] range silently
// WRAPPED them (default AP_WRAP), producing errors of exactly 2.0 for
// values like -1.125 -> 0.875. Fixed with a dedicated, correctly-sized type.
typedef ap_fixed<8, 2> weight_fc2_t;  // Q2.6: range [-2, 1.984375]

// AXI-Stream-legal output type. acc_t (ap_fixed<20,8,...>, used for all
// internal accumulation/arithmetic) is only 20 bits wide -- AXI DMA
// requires stream data width to be one of 8/16/32/64/128/256/512/1024
// bits, and 20 isn't in that set (every other stream in this project
// carries spike_t/pooled_t, narrow enough that Vitis HLS auto-rounds them
// up to 8 bits, which happens to already be DMA-legal -- acc_t is the
// only type in this design wide enough to fall outside that safety net,
// which only surfaced once the design was actually wired into a real
// axi_dma IP in Vivado, not during any earlier HLS-only verification).
// Fixed by widening ONLY the stream port to a DMA-legal 32-bit type via a
// lossless cast at the point of writing -- accum[] and all internal
// arithmetic remain acc_t, unchanged and already verified.
typedef ap_fixed<32, 8, AP_RND_CONV, AP_SAT> axis_score_t;

// Packet type for class_scores_out: ap_axiu<32,0,0,0> bundles TDATA (32
// bits) with an explicit TLAST bit (and TKEEP/TSTRB, auto-managed). This
// is required because AXI DMA's S2MM channel needs TLAST to know where
// one complete transfer ends -- a plain hls::stream<axis_score_t> gives
// Vitis HLS no explicit source to generate TLAST from, so it's simply
// omitted from the generated port entirely (not just left unconnected --
// the pin doesn't exist), which is exactly the warning that surfaced this.
// TLAST is asserted on the 40th (last) score of every call -- i.e. once
// per timestep, matching this project's per-timestep PS orchestration
// model (the PS re-triggers every block once per timestep anyway, so a
// per-timestep DMA completion/interrupt granularity is the natural fit).
typedef ap_axiu<32, 0, 0, 0> score_packet_t;

// Declaration only. No 'inline', no logic body -- required for any top
// function using hls::stream<T>.
//
// fc2 is the READOUT layer: no LIF, no threshold, no spike, no per-timestep
// reset. It's a pure accumulator -- sums weighted votes across all T
// timesteps and reports a real-valued score per class, not a spike.
//
// clear_accum: asserted (1) by the PS on the FIRST timestep of a NEW
//   image's inference, to zero the accumulator before that timestep's
//   contribution is added. Deasserted (0) for every subsequent timestep
//   within the same image, so contributions keep accumulating.
// class_scores_out: the CURRENT running total is written out on every
//   call. The PS only needs to read the value after the LAST timestep
//   (t=T-1) -- intermediate reads are valid running totals but not the
//   final classification result.
//
// Input is fc1_layer_axi's spike output (spike_t, single-bit) -- NOT a
// pooled multi-level value. This means, like conv1 and UNLIKE conv2/fc1,
// fc2's MAC needs no real multiplier at all (a multiply against a single
// bit degenerates to a conditional accumulate). No time-multiplexing
// (no FC2_GROUP) is needed here -- the full 64-tap MAC per neuron can run
// fully parallel in one cycle, same efficient pattern as conv1.
void fc2_layer_axi(
    hls::stream<spike_t> &fc1_spikes_in,
    const weight_fc2_t weights[FC2_OUT][FC2_IN],
    ap_uint<1> clear_accum,
    hls::stream<score_packet_t> &class_scores_out
);

#endif
