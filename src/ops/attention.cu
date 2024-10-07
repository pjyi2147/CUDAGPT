#include "ops/attention.h"
#include "ops/matMul.h"
#include "ops/matScale.h"
#include "ops/matMask.h"
#include "ops/softmax.h"
#include "ops/transpose.h"
#include <cassert>

__global__ void d_attention() {

}

void attention(float *d_Q, float *d_K, float *d_V, float *d_out, int rows, int cols, bool masked = false) {
    transpose(d_K, d_K, rows, cols);
    matMul(d_Q, d_K, d_out, rows, cols, rows);
    matScale(d_out, d_out, 1/sqrtf(cols), rows, rows);
    if (masked) {
        futureMask(d_out, seq_len, d_model);
    }
    softmax(d_out, d_out, rows, rows);
    matMul(d_out, d_V, d_out, rows, rows, cols);

}

void attention_test() {

}
