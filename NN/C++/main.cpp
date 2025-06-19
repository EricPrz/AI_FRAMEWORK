#include <iostream>
#include "tensor.h"
#include "layer.h"
#include "optimizer.h"
#include "loss.h"

int main() {
    int input_shape[] = {1, 28*28};
    Tensor* input = Tensor::randoms(input_shape, 2, 0, 1);
    input->print();

    Linear* lin1 = new Linear(28*28, 10);

    

    std::vector<std::vector<Tensor*>> params = {{lin1->weights, lin1->bias}};
    SGD* optimizer = new SGD(params, 0.00001); // Learning rate = 0.01
    CrossEntropy* loss_fn = new CrossEntropy();

    int out_shape[] = {1, 10};
    

    for (int i = 0; i<5; i++){
        // Tensor* output = lin1->forward(input);
        Tensor* output = lin1->forward(input);

        Tensor* loss = loss_fn->compute(output, Tensor::zeros(out_shape, 2));
        std::cout << "Loss: ";
        loss->print();
        std::cout << std::endl;

        loss->backward();
        
        std::cout << "Output: ";
        output->print();
        std::cout << std::endl;

        optimizer->step();
    }

    delete input;
    delete lin1;
    delete optimizer;

    return 0;
}

