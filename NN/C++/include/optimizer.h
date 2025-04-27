// optimizer.h
#pragma once

#include <vector>
#include "tensor.h"  // We need Tensor for parameters


class Optimizer {
public:
    std::vector<std::vector<Tensor*>> parameters;
    Optimizer(std::vector<std::vector<Tensor*>> params);

    virtual ~Optimizer();

    virtual void step() = 0; // pure virtual (forces subclass to implement)
};

class SGD : public Optimizer {
public:
    float lr;
    bool autoZero;

    SGD(std::vector<std::vector<Tensor*>> parameters, float lr = 0.01, bool autoZero = true);
    ~SGD();

    void step() override;
};
