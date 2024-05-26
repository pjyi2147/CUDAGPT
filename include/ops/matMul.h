#ifndef MATMUL_H
#define MATMUL_H

#include "block_size.h"

void matMul(float *d_A, float *d_B, float *d_C, int M, int N);
void matMul_test();

#endif /* MATMUL_H */
