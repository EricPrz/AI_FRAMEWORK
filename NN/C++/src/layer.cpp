#include "layer.h"

Linear::Linear(int in_features, int out_features, bool bias) : hasBias(bias) {
    int weightShape[2] = {in_features, out_features};
    this->weights = Tensor::randoms(weightShape, 2, -0.5f, 0.5f);

    if (bias) {
        int biasShape[2] = {1, out_features};
        this->bias = Tensor::zeros(biasShape, 2);
    } else {
        this->bias = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    // matmul returns a shared_ptr<Tensor>
    auto out = input->matmul(this->weights);

    if (this->hasBias && this->bias != nullptr) {
        out = *out + this->bias;  // operator+ returns shared_ptr<Tensor>
    }

    return out;
}