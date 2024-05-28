#include <iostream>
#include "ops/layernorm.h"
#include "ops/matMul.h"
#include "ops/matScale.h"
#include "ops/relu.h"
#include "ops/softmax.h"
#include "ops/transpose.h"
#include "ops/vectAdd.h"
#include "model/linear.h"

int main() {
    // tests
    layernorm_test();
    matMul_test();
    matScale_test();
    relu_test();
    transpose_test();
    softmax_test();
    vectAdd_test();
    vectBatchAdd_test();
    linear_test();
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
