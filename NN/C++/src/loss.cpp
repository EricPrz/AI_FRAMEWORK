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
    loss = loss->power(2);
    return loss; 
}
