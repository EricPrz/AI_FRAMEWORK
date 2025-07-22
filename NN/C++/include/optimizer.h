// optimizer.h
#pragma once

#include <vector>
#include <memory>
#include "tensor.h"  // We need Tensor for parameters


class Optimizer {
public:
    // Parameters: vector of vector of shared_ptr<Tensor>
    std::vector<std::vector<std::shared_ptr<Tensor>>> parameters;

    // Constructor accepts parameters by const reference to avoid unnecessary copies
    explicit Optimizer(const std::vector<std::vector<std::shared_ptr<Tensor>>>& params);

    virtual ~Optimizer() = default; // default virtual destructor

    virtual void step() = 0; // pure virtual method, must be implemented by subclasses
};


class SGD : public Optimizer {
public:
    float lr;
    bool autoZero;

    // Use member initializer list, provide default values
    SGD(const std::vector<std::vector<std::shared_ptr<Tensor>>>& params, float learning_rate = 0.01f, bool auto_zero = true);

    ~SGD() override = default;

    void step() override;
};