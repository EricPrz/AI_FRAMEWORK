// loss.cpp
#include "loss.h"
#include <numeric>
#include <cmath>

Loss::Loss() {
    loss = nullptr;
}

Loss::~Loss() {}

MSE::MSE() : Loss() {
    loss = nullptr;
}

MSE::~MSE(){}

Tensor* MSE::compute(Tensor* x, Tensor* y) {
    loss = *x - y;
    float* pow = new float;
    *pow = 2;
    loss = loss->power(pow);
    return loss; 
}



CrossEntropy::CrossEntropy() : Loss() {
    loss = nullptr;
}

CrossEntropy::~CrossEntropy(){}

Tensor* CrossEntropy::compute(Tensor* x, Tensor* y) {
    // loss = *x - y;
    // float* pow = new float;
    // *pow = 2;
    // loss = loss->power(pow);
    // return loss;

    Tensor *softmax = x->softmax();
    return softmax;

    loss = *softmax - y;

    return loss;

}


