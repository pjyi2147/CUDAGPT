// #include <cuda_runtime.h>
// #include <math.h>


// __device__ void scale(float* output, int seq_len, int d_model)
// {
//     float scale = 1.0f / sqrtf(d_model);
//     int index = blockIdx.x * d_model + threadIdx.x;
//     output[index] *= scale;
// }

// __device__ void mask(float* output, int seq_len, int d_model)
// {
//     int index = blockIdx.x * d_model + threadIdx.x;
//     if (threadIdx.x >= blockIdx.x) {
//         output[index] = -INFINITY;
//     }
// }

// __global__ void multi_head_attention(float* Q, float* K, float* V, float* output, int seq_len, int d_model, float num_heads, bool masked)
// {
//     matmul<<<seq_len, d_model>>>(Q, K, output, seq_len, d_model);
//     cudaDeviceSynchronize();

//     scale(output, seq_len, d_model);
//     cudaDeviceSynchronize();

//     if (masked) {
//         mask(output, seq_len, d_model);
//         cudaDeviceSynchronize();
//     }

//     softmax<<<seq_len, d_model>>>(output, seq_len, d_model);
//     cudaDeviceSynchronize();

//     matmul<<<seq_len, d_model>>>(output, V, output, seq_len, d_model);
//     cudaDeviceSynchronize();
// }
