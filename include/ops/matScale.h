#ifndef MATSCALE_H
#define MATSCALE_H

#include "block_size.h"

void matScale(float *d_A, float *d_B, float scale, int M, int N);
void matScale_test();

#endif /* MATSCALE_H */
