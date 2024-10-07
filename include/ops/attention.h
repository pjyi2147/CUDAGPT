#ifndef ATTENTION_H
#define ATTENTION_H

#include "block_size.h"

void attention(float *d_Q, float *d_K, float *d_V, int rows, int cols);
void attention_test();

#endif /* ATTENTION_H */
