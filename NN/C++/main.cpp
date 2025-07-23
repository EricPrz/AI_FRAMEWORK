#include <iostream>
#include <memory>
#include "tensor.h"
#include "layer.h"
#include "optimizer.h"
#include "loss.h"

#define DEBUG 0

int batch_size = 1;
int epochs = 5;

int main() {
    if (DEBUG){
        std::cout << "Debugging..." << std::endl;
    }

    int input_shape[] = {batch_size, 28 * 28};
    auto input = Tensor::randoms(input_shape, 2, 0.0f, 0.4f);

    auto lin1 = std::make_shared<Linear>(28 * 28, 28 * 28);
    auto lin2 = std::make_shared<Linear>(28 * 28, 10);

    std::vector<std::vector<std::shared_ptr<Tensor>>> params = {
        {lin1->weights, lin1->bias},
        {lin2->weights, lin2->bias}
    };

    auto optimizer = std::make_shared<SGD>(params, 0.001f); // Learning rate
    auto loss_fn = std::make_shared<CrossEntropy>();

    int out_shape[] = {batch_size, 10};

    for (int j = 1; j <= epochs; j++){

        std::cout << "Epoch " << j << ":" << std::endl;

        for (int i = 0; i < 60000; ++i) {
            auto l1 = lin1->forward(input);
            auto output = lin2->forward(l1);

            if (DEBUG){
                std::cout << "Prediction Computed" << std::endl;
            }

            auto label = Tensor::zeros(out_shape, 2);
            label->data[4] = 1.0f;

            auto loss = loss_fn->compute(output, label);

            if (DEBUG){
                std::cout << "Loss Computed" << std::endl;
            }


            if (i % 1000 == 0){

                std::cout << "Loss: ";
                loss->print();

                loss->backward();

                if (DEBUG){
                    std::cout << "Backwarded" << std::endl;
                }


                std::cout << "Output: ";
                output->print();
            }

            optimizer->step();

            if (DEBUG){
                std::cout << "Optim Step Computed" << std::endl;
            }

        }
    }

    return 0;
}
