// SpikeFace - Checkpoint 2: shared fixed-point type definitions
//
// Precision chosen from the Checkpoint 2 quantization sweep (quantization_sweep.py):
// 8-bit gave 93.75% accuracy at T=50, matching float32, using only 11.7% of the
// PYNQ-Z2 BRAM budget. Weight format is Q1.7 (1 integer bit incl. sign, 7 fractional)
// for conv1, matching the auto-selected int_bits=1 found for that layer in the sweep.
// Membrane potential and threshold use Q3.5 (3 integer bits, 5 fractional) --
// see MEM_INT_BITS in quantization_sweep.py.
#ifndef SPIKEFACE_TYPES_H
#define SPIKEFACE_TYPES_H

#include "ap_fixed.h"
#include "ap_int.h"

// Weight type: Q1.7 (matches conv1's auto-selected int_bits=1 at 8-bit total width)
typedef ap_fixed<8, 1> weight_t;

// Spike type: a spike is exactly 0 or 1, so it is a true single unsigned bit --
// NOT ap_fixed<8,1> (signed, 1 integer bit), whose representable range is
// [-1, 0.9921875] and therefore CANNOT hold the value 1.0 (it wraps to -1.0,
// flipping the sign of every MAC term that touches an active spike). Using
// ap_uint<1> is both the numerically correct choice and the cheaper one in
// hardware: gating on a spike degenerates the "multiply" into a conditional
// accumulate, so no real multiplier is needed for the spike input at all.
typedef ap_uint<1> spike_t;

// Membrane potential / threshold / leak type: Q3.5
//
// Quantization mode: AP_RND_CONV, not AP_RND. This was found via the
// full-array verification (tb_conv1_array.cpp): AP_RND is Xilinx's
// "round half up" mode, but the Python reference uses torch.round(), which
// is round-half-to-even (banker's rounding) -- matched here by AP_RND_CONV
// ("convergent rounding"). The mismatch was subtle: single-neuron testing
// with only 20 timesteps never happened to land on an exact tie value, so
// it passed bit-exact; only the full 8192-neuron array surfaced the ~1%
// of neurons whose membrane potential landed exactly on a rounding
// boundary. Worth documenting as a case where small-scale unit tests can
// pass while a scaled-up design still has a latent bug.
typedef ap_fixed<8, 3, AP_RND_CONV, AP_SAT> mem_t;

// Accumulator for MAC results before quantization back down to mem_t.
// Wider than mem_t to avoid intermediate overflow during a 5x5=25-tap MAC.
typedef ap_fixed<20, 8, AP_RND_CONV, AP_SAT> acc_t;

#endif
