#ifndef MATMUL_H
#define MATMUL_H

#include "block_size.h"

void matMul(const float *d_A, const float *d_B, float *d_C, int M, int N, int O);
void matMul_test();

#endif /* MATMUL_H */
