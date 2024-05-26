#ifndef DOTPROD_H
#define DOTPROD_H

#include "block_size.h"

void dotProd(float *d_a, float *d_b, float *d_dotprod, float *prodVec, int N);
void dotProd_test();


#endif /* DOTPROD_H */
