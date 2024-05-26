#ifndef MATADD_H
#define MATADD_H

#include "block_size.h"

void matAdd(float *d_a, float *d_b, float *d_c, int rows, int cols);
void matAdd_test();

#endif /* MATADD_H */
