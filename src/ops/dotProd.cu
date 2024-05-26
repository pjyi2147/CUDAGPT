#include "ops/dotProd.h"

__global__ void d_dotProd(float *d_a, float *d_b, float *d_dotprod, int N) {
    //TODO: implement dotprod
}

void dotProd(float *d_a, float *d_b, float *d_dotprod, float *prodVec, int N)
{
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    d_dotProd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_dotprod, N);
}

void dotProd_test()
{

}
