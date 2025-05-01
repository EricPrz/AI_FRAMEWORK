// loss.cpp
#include "loss.h"

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
