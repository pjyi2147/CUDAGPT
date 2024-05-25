

// void decoder_layer(float* Q, float* K, float* V, float* output, float* gamma, float* beta, int seq_len, int d_model, int num_heads)
// {
//     multi_head_attention<<<seq_len, d_model>>>(Q, K, V, output, seq_len, d_model, num_heads);
//     cudaDeviceSynchronize();

//     add<<<seq_len, d_model>>>(output, Q, seq_len, d_model);
//     cudaDeviceSynchronize();

//     layer_norm<<<seq_len, d_model>>>(output, gamma, beta, seq_len, d_model);
//     cudaDeviceSynchronize();

//     feed_forward<<<seq_len, d_model>>>(output, gamma, beta, seq_len, d_model);
//     cudaDeviceSynchronize();

//     add<<<seq_len, d_model>>>(output, Q, seq_len, d_model);
//     cudaDeviceSynchronize();

//     layer_norm<<<seq_len, d_model>>>(output, gamma, beta, seq_len, d_model);
//     cudaDeviceSynchronize();
// }
