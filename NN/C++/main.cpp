#include <iostream>
#include "tensor.h"
#include "layer.h"
#include "optimizer.h"
#include "loss.h"

int main() {
    int input_shape[] = {1, 4};
    Tensor* input = Tensor::ones(input_shape, 2);

    Linear* lin1 = new Linear(4, 1);

    

    std::vector<std::vector<Tensor*>> params = {{lin1->weights, lin1->bias}};
    SGD* optimizer = new SGD(params, 0.01); // Learning rate = 0.01
    MSE* loss_fn = new MSE();

    int out_shape[] = {1, 1};
    

    for (int i = 0; i<5; i++){
        Tensor* output = lin1->forward(input);
        std::cout << "Output: ";
        output->print();

        Tensor* loss = loss_fn->compute(output, Tensor::zeros(out_shape, 2));
        std::cout << "Loss: ";
        loss->print();
        loss->backward();
        lin1->weights->gradient->print();

        optimizer->step();
    }

    delete input;
    delete lin1;
    delete optimizer;

    return 0;
}

