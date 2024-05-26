#ifndef BMM_H
#define BMM_H

#include "block_size.h"

void bmm(float *d_A, float *d_B, float *d_C, int batch_size, int M, int N, int P);
void bmm_test();

#endif /* BMM_H */
