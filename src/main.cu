#include <iostream>
#include "kernel.h"

int main() {
    // Example array
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;

    // Allocate host memory
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Print some results
    std::cout << "y[0] = " << y[0] << std::endl;
    std::cout << "y[N-1] = " << y[N-1] << std::endl;

    // Free host memory
    free(x);
    free(y);

    return 0;
}





// // Implement GPT2 model in CUDA C++.


// int main() {
// // 1. take input

// // 2. Input embedding

// // 3. Positional encoding

// // 4. Transformer

// // 5. Linear

// // 6. Softmax

// // 7. Output


//     return 0;
// }
