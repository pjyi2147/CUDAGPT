#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "block_size.h"

void transpose(float *d_A, float *d_T, int M, int N);
void transpose_test();

#endif /* TRANSPOSE_H */
