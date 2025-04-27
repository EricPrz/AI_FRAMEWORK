// layer.cpp
#include "layer.h"

Layer::Layer() {
    parameters = nullptr;
}

Layer::~Layer() {}

Linear::Linear(int in_features, int out_features, bool bias) {
    this->hasBias = bias;
    int weightShape[2] = {in_features, out_features};
    this->weights = Tensor::randoms(weightShape, 2);

    if (bias) {
        int biasShape[2] = {1, out_features};
        this->bias = Tensor::zeros(biasShape, 2);
    } else {
        this->bias = nullptr;
    }
}

Linear::~Linear() {
    delete weights;
    if (bias != nullptr) delete bias;
}

Tensor* Linear::forward(Tensor* input) {
    Tensor* out = input->matmul(this->weights);
    if (this->hasBias && this->bias != nullptr) {
        out = *out + this->bias;
    }
    return out;
}
