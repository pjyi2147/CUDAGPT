#ifndef RELU_H
#define RELU_H

#include "block_size.h"

void relu(const float* d_in, float* d_out, int rows, int cols);
void relu_test();

#endif /* RELU_H */
