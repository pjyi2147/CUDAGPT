#include "model/MHA.h"
#include <ops/matMul.h>
#include <ops/matScale.h>
#include <ops/softmax.h>

class MHA {
public:
    static void forward(float* d_Q, float* d_K, float* d_V, float* d_output,
                        int seq_len, int d_model, float num_heads, bool masked) {
    }
};
