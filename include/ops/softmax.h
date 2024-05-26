#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "block_size.h"

void softmax(const float* d_in, float* d_out, int rows, int cols);
void softmax_test();

#endif /* SOFTMAX_H */
