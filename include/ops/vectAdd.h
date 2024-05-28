#ifndef VECTADD_H
#define VECTADD_H

#include "block_size.h"

void vectAdd(float *d_A, float *d_T, int M, int N);
void vectAdd_test();

void vectBatchAdd(float *d_in, float *d_add, float *d_out, int M, int N);
void vectBatchAdd_test();

#endif /* VECTADD_H */
