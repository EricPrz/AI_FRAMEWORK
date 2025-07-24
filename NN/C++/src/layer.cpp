#include "layer.h"
#include <vector>

Linear::Linear(int in_features, int out_features, bool bias) : hasBias(bias) {
    std::vector<int> weightShape = {in_features, out_features};
    this->weights = Tensor::randoms(weightShape, -0.5f, 0.5f);

    if (bias) {
        std::vector<int> biasShape = {1, out_features};
        this->bias = Tensor::zeros(biasShape);
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
