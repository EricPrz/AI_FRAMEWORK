// optimizer.cpp
#include "optimizer.h"

Optimizer::Optimizer(const std::vector<std::vector<std::shared_ptr<Tensor>>>& params)
    : parameters(params) {}

SGD::SGD(const std::vector<std::vector<std::shared_ptr<Tensor>>>& params, float lr, bool autoZero)
    : Optimizer(params), lr(lr), autoZero(autoZero) {}

void SGD::step() {
    for (auto& layerParams : parameters) {
        for (auto& param : layerParams) {
            if (param->gradient) {
                for (int i = 0; i < param->size; i++) {
                    param->data[i] -= lr * param->gradient->data[i];
                    if (autoZero) {
                        param->gradient->data[i] = 0.0f;
                    }
                }
            }
        }
    }
}
