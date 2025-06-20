// loss.cpp
#include "loss.h"
#include <numeric>
#include <cmath>
#include <iostream>

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
    Tensor *softmax = x->softmax();

    loss = *softmax - y;

    float* log = new float[1];
    log[0] = 0;
    for (int i = 0; i < softmax->size; i++){
        log[0] -= y->data[i] * std::log(softmax->data[i]); 
    }

    int *shape = new int[1];
    shape[0] = 1;

    Tensor* crossentropyloss = new Tensor(log, shape, 1, x->requires_grad);
    crossentropyloss->set_creator(softmax, y, "crossentropy");

    return crossentropyloss;

}


