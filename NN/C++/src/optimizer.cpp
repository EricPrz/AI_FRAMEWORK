// optimizer.cpp
#include "optimizer.h"

Optimizer::Optimizer(std::vector<std::vector<Tensor*>> params) {
    this->parameters = params;
}

Optimizer::~Optimizer() {}

SGD::SGD(std::vector<std::vector<Tensor*>> params, float lr, bool autoZero) : Optimizer(params) {
    this->lr = lr;
    this->autoZero = autoZero;
}

SGD::~SGD() {}

void SGD::step() {
    for (auto& layerParams : parameters) {
        for (auto& param : layerParams) {
            if (param->gradient != nullptr) {
                for (int i = 0; i < param->size; i++) {
                    param->data[i] -= lr * param->gradient->data[i];
                    if (autoZero) {
                        param->gradient->data[i] = 0;
                    }
                }
            }
        }
    }
}
