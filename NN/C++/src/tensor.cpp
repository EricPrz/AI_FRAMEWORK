// tensor.cpp
#include "tensor.h"
#include <random>
#include <cstring> // for memcpy
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

Tensor::Tensor(std::shared_ptr<float[]> dataPtr, int shape[], int dims, bool requires_grad) {
    this->size = 1;
    this->dims = dims;
    this->shape = std::shared_ptr<int[]>(new int[dims], std::default_delete<int[]>());
    for (int i = 0; i < dims; i++) {
        this->size *= shape[i];
        this->shape[i] = shape[i];
    }
    this->data = dataPtr;
    this->gradient = nullptr;
    this->requires_grad = requires_grad;
    this->creator_a = nullptr;
    this->creator_b = nullptr;
    this->creation_op = "";
}

Tensor::~Tensor() {
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
    if (this->dims != 2 || other->dims != 2 || this->shape[1] != other->shape[0]) {
        throw std::runtime_error("Invalid shapes for matmul");
    }
    
    int m = this->shape[0];
    int n = this->shape[1];
    int p = other->shape[1];

    std::shared_ptr<float[]> result(new float[m * p]());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i*p + j] += this->data[i*n + k] * other->data[k*p + j];
            }
        }
    }

    std::shared_ptr<int[]> newShape(new int[2]{m, p});
    auto tns = std::make_shared<Tensor>(result, newShape.get(), 2, this->requires_grad || other->requires_grad);
    tns->set_creator(shared_from_this(), other, "matmul");
    return tns;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    
    if (this->size != other->size && this->size % other->size != 0 && other->size % this->size != 0) {
        std::cout << "Error: " << this->size << " " << other->size << std::endl;
        throw std::runtime_error("Size mismatch in tensor addition\n");
    }

    std::shared_ptr<float[]> result(new float[this->size]);
    
    for (int i = 0; i < this->size; i++) {
        result[i] = this->data[i] + other->data[i];
    }

    std::shared_ptr<int[]> newShape(new int[this->dims]);
    for (int i = 0; i < this->dims; i++) {
        newShape[i] = this->shape[i];
    }

    auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad || other->requires_grad);
    tns->set_creator(shared_from_this(), other, "add");
    return tns;


    // if (this->size == other->size){
    //
    //     std::shared_ptr<float[]> result(new float[this->size]);
    //
    //     for (int i = 0; i < this->size; i++) {
    //         result[i] = this->data[i] + other->data[i];
    //     }
    //
    //     std::shared_ptr<int[]> newShape(new int[this->dims]);
    //     for (int i = 0; i < this->dims; i++) {
    //         newShape[i] = this->shape[i];
    //     }
    //
    //     auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad || other->requires_grad);
    //     tns->set_creator(shared_from_this(), other, "add");
    //     return tns;
    // }
    //
    // else if (this->size % other->size == 0){
    //
    //     std::shared_ptr<float[]> result(new float[this->size]);
    //
    //     for (int i = 0; i < this->size; i++) {
    //         result[i] = this->data[i] + other->data[i % other->size];
    //     }
    //
    //     std::shared_ptr<int[]> newShape(new int[this->dims]);
    //     for (int i = 0; i < this->dims; i++) {
    //         newShape[i] = this->shape[i];
    //     }
    //
    //     auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad || other->requires_grad);
    //     tns->set_creator(shared_from_this(), other, "add");
    //     return tns;
    // }
    //
    // else {
    //
    //     std::shared_ptr<float[]> result(new float[other->size]);
    //
    //     for (int i = 0; i < other->size; i++) {
    //         result[i] = other->data[i] + this->data[i % this->size];
    //     }
    //
    //     std::shared_ptr<int[]> newShape(new int[other->dims]);
    //     for (int i = 0; i < other->dims; i++) {
    //         newShape[i] = other->shape[i];
    //     }
    //
    //     auto tns = std::make_shared<Tensor>(result, newShape.get(), other->dims, this->requires_grad || other->requires_grad);
    //     tns->set_creator(shared_from_this(), other, "add");
    //     return tns;
    // }
    //
    // // Should never reach here, but just in case:
    // throw std::runtime_error("Unexpected case in tensor addition logic");

}

std::shared_ptr<Tensor> Tensor::operator-() {
    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < this->size; i++) {
        result[i] = - this->data[i];
    }

    std::shared_ptr<int[]> newShape(new int[this->dims]);
    for (int i = 0; i < this->dims; i++) {
        newShape[i] = this->shape[i];
    }

    // Create a new Tensor shared_ptr using make_shared for exception safety and clarity
    auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad);

    // Set creator for autograd graph (nullptr for creator_b since unary op)
    tns->set_creator(shared_from_this(), nullptr, "subtract");

    return tns;
}

std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other) {

    if (this->size != other->size) {
        std::cout << "Error: " << this->size << " " << other->size << std::endl;
        throw std::runtime_error("Size mismatch in tensor substraction\n");
    }

    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < this->size; i++) {
        result[i] = this->data[i] - other->data[i];
    }

    std::shared_ptr<int[]> newShape(new int[this->dims]);
    for (int i = 0; i < this->dims; i++) {
        newShape[i] = this->shape[i];
    }

    auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad || other->requires_grad);

    tns->set_creator(shared_from_this(), other, "subtract");

    return tns;
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    // Check if shapes match element-wise
    bool shapes_equal = (this->dims == other->dims);
    for (int i = 0; shapes_equal && i < this->dims; i++) {
        if (this->shape[i] != other->shape[i]) {
            shapes_equal = false;
        }
    }

    if (shapes_equal) {
        std::shared_ptr<float[]> result(new float[this->size]);
        for (int i = 0; i < this->size; i++) {
            result[i] = this->data[i] * other->data[i];
        }

        std::shared_ptr<int[]> newShape(new int[this->dims]);
        for (int i = 0; i < this->dims; i++) {
            newShape[i] = this->shape[i];
        }

        auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad || other->requires_grad);
        tns->set_creator(shared_from_this(), other, "mul");
        return tns;
    } 
    else if (this->size == 1) {
        // Scalar multiplication: this tensor is scalar
        std::shared_ptr<float[]> result(new float[other->size]);
        for (int i = 0; i < other->size; i++) {
            result[i] = this->data[0] * other->data[i];
        }

        std::shared_ptr<int[]> newShape(new int[other->dims]);
        for (int i = 0; i < other->dims; i++) {
            newShape[i] = other->shape[i];
        }

        auto tns = std::make_shared<Tensor>(result, newShape.get(), other->dims, this->requires_grad || other->requires_grad);
        tns->set_creator(shared_from_this(), other, "mul");
        return tns;
    }
    else {
        throw std::runtime_error("Tensor shape mismatch: cannot multiply tensors with incompatible shapes");
    }
}

std::shared_ptr<Tensor> Tensor::transpose() {
    if (this->dims != 2) {
        throw std::runtime_error("Transpose only supports 2D tensors.");
    }

    int rows = this->shape[0];
    int cols = this->shape[1];
    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = this->data[i * cols + j];
        }
    }

    auto newShape = std::make_shared<int[]>(2);
    newShape[0] = cols;
    newShape[1] = rows;

    auto tns = std::make_shared<Tensor>(result, newShape.get(), 2, this->requires_grad);
    tns->set_creator(shared_from_this(), nullptr, "transpose");

    return tns;
}

std::shared_ptr<Tensor> Tensor::power(std::shared_ptr<float> power) {
    // Allocate result array managed by shared_ptr
    std::shared_ptr<float[]> result(new float[this->size]);

    int rows = this->shape[0];
    int cols = this->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = std::pow(this->data[i * cols + j], *power);
        }
    }

    // Copy shape to a new shared_ptr<int[]>
    std::shared_ptr<int[]> newShape(new int[this->dims]);
    for (int i = 0; i < this->dims; i++) {
        newShape[i] = this->shape[i];
    }

    // Create the power Tensor argument (scalar)
    int powerShape[1] = {1};
    std::shared_ptr<float[]> powerData(new float[1]{*power});
    auto powerTns = std::make_shared<Tensor>(powerData, powerShape, 1);

    // Create the result tensor with requires_grad = this->requires_grad
    auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad);

    // Set creators for autograd graph
    tns->set_creator(shared_from_this(), powerTns, "power");

    return tns;
}

std::shared_ptr<Tensor> Tensor::softmax() {
    // Allocate result array with shared_ptr
    std::shared_ptr<float[]> result(new float[this->size]);

    float total = 0.0f;

    for (int i = 0; i < this->size; i++) {
        result[i] = std::exp(this->data[i]);
        total += result[i];
    }

    for (int i = 0; i < this->size; i++) {
        result[i] /= total;
    }

    // Copy shape to new shared_ptr<int[]>
    std::shared_ptr<int[]> newShape(new int[this->dims]);
    for (int i = 0; i < this->dims; i++) {
        newShape[i] = this->shape[i];
    }

    // Create the new Tensor with requires_grad inherited
    auto tns = std::make_shared<Tensor>(result, newShape.get(), this->dims, this->requires_grad);

    // Set creator for autograd graph; second creator nullptr since unary op
    tns->set_creator(shared_from_this(), nullptr, "softmax");

    return tns;
}

void Tensor::set_creator(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, const std::string& op) {
    this->creator_a = a;
    this->creator_b = b;
    this->creation_op = op;
}

void Tensor::backward() {
    if (!this->requires_grad) return;

    // Initialize gradient to ones if it doesn't exist
    if (this->gradient == nullptr) {
        auto shapeCopy = std::make_shared<int[]>(this->dims);
        std::memcpy(shapeCopy.get(), this->shape.get(), this->dims * sizeof(int));
        this->gradient = Tensor::ones(shapeCopy.get(), this->dims);
    }


    if (this->creator_a != nullptr) {
        if (creation_op == "add") {
            this->creator_a->gradient = this->gradient;
            this->creator_b->gradient = this->gradient;
            this->creator_a->backward();
            this->creator_b->backward();
        }
        else if (creation_op == "substract") {
            this->creator_a->gradient = this->gradient;
            this->creator_b->gradient = -(*this->gradient);
            this->creator_a->backward();
            this->creator_b->backward();
        }
        else if (creation_op == "power") {
            this->creator_a->gradient = *this->creator_b * this->gradient;
            this->creator_a->backward();
        }
        else if (creation_op == "matmul") {
            auto grad_a = this->gradient->matmul(this->creator_b->transpose());
            auto grad_b = this->creator_a->transpose()->matmul(this->gradient);

            this->creator_a->gradient = grad_a;
            this->creator_b->gradient = grad_b;

            this->creator_a->backward();
            this->creator_b->backward();
        }
        else if (creation_op == "mul"){
            this->creator_a->gradient = *this->creator_b * this->gradient;
            this->creator_b->gradient = *this->creator_a * this->gradient;
            this->creator_a->backward();
            this->creator_b->backward();
        }
        else if (creation_op == "crossentropy"){
            // Derivative: softmax - y
            if (this->creator_a && this->creator_a->creator_a) {
                this->creator_a->creator_a->gradient = *this->creator_a - this->creator_b;
                this->creator_a->creator_a->backward();
            }
        }
    }
}

void Tensor::print() {
    if (dims == 2) {
        int rows = shape[0];
        int cols = shape[1];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        for (int i = 0; i < size; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
}

std::shared_ptr<Tensor> Tensor::zeros(int shape[], int dims) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Use shared_ptr<float[]> to manage the float array
    std::shared_ptr<float[]> arr(new float[size]());

    // Use make_shared for Tensor to ensure exception safety
    return std::make_shared<Tensor>(arr, shape, dims);
}

std::shared_ptr<Tensor> Tensor::ones(int shape[], int dims) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Allocate and initialize with 1.0f
    std::shared_ptr<float[]> arr(new float[size]);
    for (int i = 0; i < size; i++) arr[i] = 1.0f;

    // Use make_shared for exception safety
    return std::make_shared<Tensor>(arr, shape, dims);
}

std::shared_ptr<Tensor> Tensor::randoms(int shape[], int dims, float min, float max) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Use shared_ptr with array deleter
    std::shared_ptr<float[]> arr(new float[size]);

    // Setup random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    return std::make_shared<Tensor>(arr, shape, dims);
}
