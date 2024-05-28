#include "model/linear.h"

class Linear {
private:
    int in_features;
    int out_features;
    float* weight;
    float* bias;

public:
    Linear(int in_features, int out_features, float* w, float* b) {
        this->in_features = in_features;
        this->out_features = out_features;

        cudaMalloc(&this->weight, in_features * out_features * sizeof(float));
        cudaMalloc(&this->bias, out_features * sizeof(float));

        if (w == nullptr) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0, 1);

            float* h_weight = new float[in_features * out_features];
            for (int i = 0; i < in_features * out_features; i++) {
                h_weight[i] = dist(gen);
            }
            cudaMemcpy(w, h_weight, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_weight;
        }
        else {
            cudaMemcpy(this->weight, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
        }

        if (b == nullptr) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0, 1);

            float* h_bias = new float[out_features];
            for (int i = 0; i < out_features; i++) {
                h_bias[i] = dist(gen);
            }
            cudaMemcpy(b, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_bias;
        }
        else {
            cudaMemcpy(this->bias, b, out_features * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    ~Linear() {
        cudaFree(weight);
        cudaFree(bias);
    }

    void forward(float* d_in, float* d_out, int M, int N) {
        assert(N == in_features);

        // d_out = d_in * weight^T + bias
        matMul(d_in, weight, d_out, M, N);
        cudaDeviceSynchronize();

        // vectBatchAdd(d_out, bias, d_out, M, out_features);
        // cudaDeviceSynchronize();
    }
};


void linear_test()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    int M = 5;
    int N = 20;
    int D = 10;

    float* in = new float[M * N];
    float* weight = new float[N * D];
    float* bias = new float[D];
    for (int i = 0; i < M * N; i++) {
        in[i] = dist(gen);
    }

    for (int i = 0; i < N * D; i++) {
        weight[i] = dist(gen);
    }

    for (int i = 0; i < D; i++) {
        bias[i] = dist(gen);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, M * N * sizeof(float));
    cudaMalloc(&d_out, M * D * sizeof(float));

    cudaMemcpy(d_in, in, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, M * D * sizeof(float));

    Linear model(N, D, weight, bias);

    model.forward(d_in, d_out, M, N);

    float out[M * D];
    cudaMemcpy(out, d_out, M * D * sizeof(float), cudaMemcpyDeviceToHost);

    float ans[M * D];
    memset(ans, 0, M * D * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            // ans[i * D + j] = bias[j];
            for (int k = 0; k < N; k++) {
                ans[i * D + j] += in[i * N + k] * weight[k * D + j];
            }
        }
    }

    // verify
    bool pass = true;
    for (int i = 0; i < M * D; i++) {
        printf("%d: %f %f\n", i, out[i], ans[i]);
        if (out[i] != ans[i]) {
            pass = false;
        }
    }

    printf("linear test: %s\n", pass ? "pass" : "fail");
    printf("linear test done\n\n");
}
