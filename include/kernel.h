#ifndef KERNEL_H
#define KERNEL_H

__global__ void saxpy(int n, float a, float *x, float *y);

#endif // KERNEL_H
