// SpikeFace - Checkpoint 2: top-level synthesizable function for one neuron,
// one timestep. This is the function that would be wrapped with an AXI-Stream
// interface for HLS synthesis (Checkpoint 3 integration work).
#ifndef SPIKEFACE_NEURON_TOP_H
#define SPIKEFACE_NEURON_TOP_H

#include "spikeface_types.h"
#include "lif_neuron.h"

// One neuron, one timestep: 5x5 event-gated MAC -> LIF update.
// mem is the neuron's persistent membrane potential (state across calls).
inline spike_t spikeface_neuron_step(
    const spike_t patch[25],
    const weight_t kernel[25],
    mem_t &mem,
    mem_t beta,
    mem_t threshold,
    int &active_taps
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    // Synthesis (Vitis HLS 2022.2, xc7z020-clg400-1) without this pragma gave
    // Latency=29 / Interval=30 cycles (i.e. NOT pipelined -- back-to-back calls
    // fully serialize). At II=1, back-to-back neuron-timestep calls overlap,
    // which is required for the full conv1 array (8192 neurons x T=50 steps)
    // to run in a workable time budget -- see CHECKPOINT2_STATUS.md.
    acc_t cur = event_gated_mac25(patch, kernel, active_taps);
    spike_t spk = lif_update(cur, mem, beta, threshold);
    return spk;
}

#endif
