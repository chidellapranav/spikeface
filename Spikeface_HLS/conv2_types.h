// SpikeFace - Checkpoint 3 (conv2): pooled activation type.
//
// Between conv1 and conv2, the Python reference applies F.avg_pool2d(spk1, 2)
// -- averaging 4 binary spikes in each 2x2 block. This is NOT a spike
// anymore: it lands on exactly one of 5 values: 0, 0.25, 0.5, 0.75, 1.0.
// spike_t (ap_uint<1>) cannot represent this -- a new type is needed.
//
// ap_ufixed<3,1>: 1 integer bit, 2 fractional bits, unsigned. This
// represents all 5 possible pooled values EXACTLY (no rounding error):
//   0    -> 0.00
//   0.25 -> 0.01 (binary fraction)
//   0.5  -> 0.10
//   0.75 -> 0.11
//   1.0  -> 1.00
// Confirmed against the Python golden data via np.unique(), which shows
// precisely these 5 values and nothing else.
#ifndef CONV2_TYPES_H
#define CONV2_TYPES_H

#include "spikeface_types.h"

typedef ap_ufixed<3, 1> pooled_t;

#endif
