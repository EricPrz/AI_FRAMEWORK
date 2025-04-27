// layer.h
#pragma once

#include "tensor.h"  // We need Tensor for parameters and inputs.

class Layer {
public:
    Tensor* parameters;

    Layer();
    virtual ~Layer();

    virtual Tensor* forward(Tensor* input) = 0; // pure virtual (forces subclasses to implement)
};

class Linear : public Layer {
public:
    bool hasBias;
    Tensor* bias;
    Tensor* weights;

    Linear(int in_features, int out_features, bool bias = true);
    ~Linear();

    Tensor* forward(Tensor* input) override;
};
