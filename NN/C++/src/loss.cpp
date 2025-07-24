#include "loss.h"
#include "tensor.h"
#include <cmath>
#include <memory>

std::shared_ptr<Tensor> MSE::compute(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    // Compute (x - y)^2
    loss = (*x - y);
    auto power_val = std::make_shared<float>(2.0f);
    loss = loss->power(power_val);
    return loss;
}

CrossEntropy::CrossEntropy(){
}

CrossEntropy::CrossEntropy(int axis){
    this->axis = axis;
}

std::shared_ptr<Tensor> CrossEntropy::compute(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    std::shared_ptr<Tensor> softmax;
    if (this->axis){
        softmax = x->softmax(this->axis);
    }
    else {
        softmax = x->softmax();
    }

    loss = *softmax - y;

    std::shared_ptr<float[]> log(new float[1]);

    log[0] = 0.0f;
    
    for (int i = 0; i < softmax->size; i++) {
        // Add a small epsilon to avoid log(0)
        float val = softmax->data[i] > 1e-12f ? softmax->data[i] : 1e-12f;
        log[0] -= y->data[i] * std::log(val);
    }

    std::vector<int> shape = {1};

    auto cross_entropy_loss = std::make_shared<Tensor>(log, std::make_shared<std::vector<int>>(shape), x->requires_grad);
    cross_entropy_loss->set_creator(softmax, y, "crossentropy");

    return cross_entropy_loss;
}
