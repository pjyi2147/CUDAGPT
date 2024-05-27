#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "block_size.h"

void layernorm(float *d_a, float *d_norm, int N, int M);
void layernorm_test();

#endif /* LAYERNORM_H */
