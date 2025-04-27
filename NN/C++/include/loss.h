// loss.h
#pragma once

#include "tensor.h"  // We need Tensor for parameters


class Loss{
public:
    Tensor* loss;
    Loss();

    virtual ~Loss();

    virtual Tensor* compute(Tensor* x, Tensor* y) = 0; // pure virtual (forces subclass to implement)
};

class MSE : public Loss {
public:
    MSE();
    ~MSE();

    Tensor* compute(Tensor* x, Tensor* y);
};
