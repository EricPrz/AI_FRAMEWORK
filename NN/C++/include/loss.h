#pragma once

#include <memory>
#include "tensor.h"  // We need Tensor for parameters

class Loss {
public:
    std::shared_ptr<Tensor> loss;

    Loss() = default;
    virtual ~Loss() = default;

    // Pure virtual method to compute loss between prediction x and target y
    virtual std::shared_ptr<Tensor> compute(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) = 0;
};

class MSE : public Loss {
public:
    MSE() = default;
    ~MSE() override = default;

    std::shared_ptr<Tensor> compute(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) override;
};

class CrossEntropy : public Loss {
public:
    int axis;
    CrossEntropy();
    CrossEntropy(int axis);
    ~CrossEntropy() override = default;

    std::shared_ptr<Tensor> compute(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) override;
};
