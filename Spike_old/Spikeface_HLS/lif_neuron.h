// SpikeFace - Checkpoint 2: LIF neuron core + event-gated MAC
//
// Two synthesizable building blocks:
//
//  1. event_gated_mac25(): computes the weighted input current for one neuron
//     from a 5x5 (25-tap) receptive field. This is the actual "event-driven"
//     claim in the project abstract: the multiply-accumulate for each tap is
//     only performed when that tap's input spike is nonzero. In dense/CNN-style
//     hardware every tap is always computed regardless of activity; here, a
//     zero input spike costs one comparison and zero DSP-consuming multiplies.
//     active_taps is exposed so the testbench/accelerator can report realized
//     sparsity (taps skipped / 25) as a hardware activity metric.
//
//  2. lif_update(): the leaky-integrate-fire state update for one neuron for
//     one timestep -- accumulate input current onto membrane potential (with
//     leak), compare to threshold, fire + reset-to-zero on threshold crossing.
//     All arithmetic is in the fixed-point types from spikeface_types.h, so
//     this is bit-accurate to the Python quantized_snn_forward() reference.
#ifndef LIF_NEURON_H
#define LIF_NEURON_H

#include "spikeface_types.h"

// Event-gated MAC over a 5x5 receptive field.
// patch/kernel are laid out row-major, 25 elements each.
// Returns the accumulated current; active_taps counts how many of the 25
// taps actually required a multiply (i.e. had a nonzero input spike).
inline acc_t event_gated_mac25(const spike_t patch[25], const weight_t kernel[25], int &active_taps) {
#pragma HLS INLINE
    acc_t acc = 0;
    active_taps = 0;
MAC_LOOP:
    for (int i = 0; i < 25; i++) {
#pragma HLS UNROLL
        // Event gating: only multiply-accumulate when the input spike is present.
        // In RTL this maps to clock-gating (or simply not enabling) the DSP
        // multiplier input register when patch[i] == 0, which is what actually
        // saves dynamic power -- the comparison itself is cheap (single LUT).
        if (patch[i] != spike_t(0)) {
            acc += acc_t(patch[i]) * acc_t(kernel[i]);
            active_taps++;
        }
    }
    return acc;
}

// Single LIF neuron state update for one timestep.
// mem is passed by reference (persistent state across timesteps).
// Returns 1 if the neuron fired this step, 0 otherwise. Reset-to-zero on fire.
inline spike_t lif_update(acc_t input_current, mem_t &mem, mem_t beta, mem_t threshold) {
#pragma HLS INLINE
    // Leak + integrate: mem = beta * mem + input_current
    acc_t leaked = acc_t(beta) * acc_t(mem);
    acc_t updated = leaked + input_current;

    mem_t mem_q = (mem_t)updated;  // quantize back down to Q3.5, matches Python's
                                    // quantize_fixed(beta_q*mem + cur, ...) step

    spike_t spk = 0;
    if (mem_q >= threshold) {
        spk = 1;
        mem = 0;           // reset-to-zero on spike
    } else {
        mem = mem_q;
    }
    return spk;
}

#endif
