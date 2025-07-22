#pragma once

#include <memory>
#include "tensor.h"  // We need Tensor for parameters and inputs.

class Layer {
public:
    // Could be a vector if layer has multiple parameters, but keep as is if single tensor
    std::shared_ptr<Tensor> parameters;

    Layer() = default;
    virtual ~Layer() = default;

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0; // pure virtual
};

class Linear : public Layer {
public:
    bool hasBias;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> weights;

    Linear(int in_features, int out_features, bool bias = true);
    ~Linear() override = default;

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};