// tensor.cpp
#include "tensor.h"
#include <random>
#include <cstring> // for memcpy
#include <iostream>
#include <memory>
#include <cmath>
#include <type_traits>
#include <vector>

Tensor::Tensor(std::shared_ptr<float[]> dataPtr, std::shared_ptr<std::vector<int>> shape, bool requires_grad) {
    this->shape = shape;
    this->data = dataPtr;
    this->gradient = nullptr;
    this->requires_grad = requires_grad;
    this->creator_a = nullptr;
    this->creator_b = nullptr;
    this->creation_op = "";
    this->creation_op_arg = 0;
    this->dims = 0;
    this->size = 1;
    for (int i = 0; i < shape->size(); i++){
        size *= shape->at(i);
        dims += 1;
    }

}

Tensor::~Tensor() {
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
    if (this->dims != 2 || other->dims != 2 || this->shape->at(1) != other->shape->at(0)) {
        throw std::runtime_error("Invalid shapes for matmul");
    }
    
    int m = this->shape->at(0);
    int n = this->shape->at(1);
    int p = other->shape->at(1);

    std::shared_ptr<float[]> result(new float[m * p]());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i*p + j] += this->data[i*n + k] * other->data[k*p + j];
            }
        }
    }

    auto newShape = std::make_shared<std::vector<int>>(std::vector<int>{m, p});
    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);
    tns->set_creator(shared_from_this(), other, "matmul");
    return tns;
}

std::shared_ptr<Tensor> Tensor::sum(int axis, bool keepdims){

    int resulting_size = this->size / this->shape->at(axis);
    int new_dims = this->dims + keepdims - 1;

    // Initialize result array to zero
    std::shared_ptr<float[]> result(new float[resulting_size]());

    if (axis == 0){
        for (int i = 0; i < this->size; i++){
            result[i % this->shape->at(axis)] += this->data[i];
        }
    }
    else if (axis == 1){
        for (int i = 0; i < this->size; i++){
            result[i / this->shape->at(axis)] += this->data[i];
        }
    }


    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < new_dims; i++){
        bool greater = axis >= i;
        newShape->push_back(this->shape->at(i + greater));
    }

    auto tns = std::make_shared<Tensor>(result, newShape, new_dims);
    tns->set_creator(shared_from_this(), nullptr, "sum", axis);
    return tns;
}


std::shared_ptr<Tensor> Tensor::expand_to(std::shared_ptr<Tensor> other) {
    if (other->size % this->size != 0) {
        throw std::runtime_error("Invalid size to expand");
    }
    
    std::shared_ptr<float[]> result(new float[other->size]());
    for (int i = 0; i < other->size; i++){
        result[i] = this->data[i % this->size];
    }

    int dims = other->dims;  // assuming you have this
    auto newShape = std::make_shared<std::vector<int>>();

    for (int i = 0; i < dims; ++i) {
        newShape->push_back(other->shape->at(i));
    }

    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);
    tns->set_creator(shared_from_this(), other, "expand_to");
    return tns;
}

std::shared_ptr<Tensor> Tensor::reduce_to(std::shared_ptr<Tensor> other) {
    if (this->size % other->size != 0) {
        throw std::runtime_error("Invalid size to reduce to");
    }
    
    std::shared_ptr<float[]> result(new float[other->size]());
    // Initialize to zero
    for (int i = 0; i < other->size; i++){
        result[i] = 0;
    }

    for (int i = 0; i < this->size; i++){
        result[i % other->size] += this->data[i];
    }

    int dims = other->dims;  // assuming you have this
    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < dims; ++i) {
        newShape->push_back(other->shape->at(i));
    }

    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);
    tns->set_creator(shared_from_this(), other, "reduce_to");
    return tns;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    std::shared_ptr<Tensor> a = shared_from_this();
    std::shared_ptr<Tensor> b = other;

    // Expand tensors if needed
    if (a->size != b->size) {
        if (a->size < b->size) {
            a = a->expand_to(b);
        } else {
            b = b->expand_to(a);
        }
    }

    // Safety check after expansion
    if (a->size != b->size) {
        throw std::runtime_error("Broadcasting failed: sizes do not match after expansion");
    }

    // Allocate result array
    std::shared_ptr<float[]> result(new float[a->size], std::default_delete<float[]>());

    // Element-wise addition
    for (int i = 0; i < a->size; ++i) {
        result[i] = a->data[i] + b->data[i];
    }

    // Copy shape from the broadcasted tensor (they now match)
    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < a->dims; ++i) {
        newShape->push_back(a->shape->at(i));
    }

    // Create output tensor
    auto tns = std::make_shared<Tensor>(result, newShape, a->requires_grad || b->requires_grad);

    // Autograd
    tns->set_creator(a, b, "add");
    return tns;
}

std::shared_ptr<Tensor> Tensor::operator-() {
    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < this->size; i++) {
        result[i] = - this->data[i];
    }

    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < this->dims; i++) {
        newShape->push_back(this->shape->at(i));
    }

    // Create a new Tensor shared_ptr using make_shared for exception safety and clarity
    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad);

    // Set creator for autograd graph (nullptr for creator_b since unary op)
    tns->set_creator(shared_from_this(), nullptr, "subtract");

    return tns;
}

std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other) {
    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < this->size; i++) {
        result[i] = this->data[i] - other->data[i];
    }

    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < this->dims; i++) {
        newShape->push_back(this->shape->at(i));
    }

    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);

    tns->set_creator(shared_from_this(), other, "subtract");

    return tns;
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    // Check if shapes match element-wise
    bool shapes_equal = (this->dims == other->dims);
    for (int i = 0; shapes_equal && i < this->dims; i++) {
        if (this->shape->at(i) != other->shape->at(i)) {
            shapes_equal = false;
        }
    }

    if (shapes_equal) {
        std::shared_ptr<float[]> result(new float[this->size]);
        for (int i = 0; i < this->size; i++) {
            result[i] = this->data[i] * other->data[i];
        }

        auto newShape = std::make_shared<std::vector<int>>();
        for (int i = 0; i < this->dims; i++) {
            newShape->push_back(this->shape->at(i));
        }

        auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);
        tns->set_creator(shared_from_this(), other, "mul");
        return tns;
    } 
    else if (this->size == 1) {
        // Scalar multiplication: this tensor is scalar
        std::shared_ptr<float[]> result(new float[other->size]);
        for (int i = 0; i < other->size; i++) {
            result[i] = this->data[0] * other->data[i];
        }

        auto newShape = std::make_shared<std::vector<int>>();
        for (int i = 0; i < other->dims; i++) {
            newShape->push_back(other->shape->at(i));
        }

        auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad || other->requires_grad);
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

    int rows = this->shape->at(0);
    int cols = this->shape->at(1);
    std::shared_ptr<float[]> result(new float[this->size]);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = this->data[i * cols + j];
        }
    }

    auto newShape = std::make_shared<std::vector<int>>();
    newShape->push_back(cols);
    newShape->push_back(rows);

    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad);
    tns->set_creator(shared_from_this(), nullptr, "transpose");

    return tns;
}

std::shared_ptr<Tensor> Tensor::power(std::shared_ptr<float> power) {
    // Allocate result array managed by shared_ptr
    std::shared_ptr<float[]> result(new float[this->size]);

    int rows = this->shape->at(0);
    int cols = this->shape->at(1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = std::pow(this->data[i * cols + j], *power);
        }
    }

    // Copy shape to a new shared_ptr<int[]>
    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < this->dims; i++) {
        newShape->push_back(this->shape->at(i));
    }

    // Create the power Tensor argument (scalar)
    std::vector<int> powerShape = {1};
    std::shared_ptr<float[]> powerData(new float[1]{*power});
    auto powerTns = std::make_shared<Tensor>(powerData, std::make_shared<std::vector<int>>(powerShape), true);

    // Create the result tensor with requires_grad = this->requires_grad
    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad);

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
    auto newShape = std::make_shared<std::vector<int>>();
    for (int i = 0; i < this->dims; i++) {
        newShape->push_back(this->shape->at(i));
    }

    // Create the new Tensor with requires_grad inherited
    auto tns = std::make_shared<Tensor>(result, newShape, this->requires_grad);

    // Set creator for autograd graph; second creator nullptr since unary op
    tns->set_creator(shared_from_this(), nullptr, "softmax");

    return tns;
}

std::shared_ptr<Tensor> Tensor::softmax(int axis) {
    // Allocate result array with shared_ptr
    std::shared_ptr<float[]> result(new float[this->size]);
    for (int i = 0; i < this->size; i++) {
        result[i] = std::exp(this->data[i]);
    }

    auto tensor = std::make_shared<Tensor>(result, this->shape);

    auto sum = tensor->sum(axis, true);

    if (axis == 0){
        for (int i = 0; i < this->size; i++){
            result[i] = result[i]/sum->data[i%sum->size];
        }
    }
    else if  (axis == 1){
        for (int i = 0; i < this->size; i++){
            result[i] = result[i]/sum->data[i/sum->size];
        }

    }

    // Create the new Tensor with requires_grad inherited
    auto tns = std::make_shared<Tensor>(result, this->shape, this->requires_grad);

    // Set creator for autograd graph; second creator nullptr since unary op
    tns->set_creator(shared_from_this(), nullptr, "softmax");

    return tns;
}


void Tensor::set_creator(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, const std::string& op, int arg) {
    this->creator_a = a;
    this->creator_b = b;
    this->creation_op = op;
    this->creation_op_arg = arg;
}

void Tensor::backward() {
    if (!this->requires_grad) return;

    // Initialize gradient to ones if it doesn't exist
    if (this->gradient == nullptr) {
        auto shapeCopy = std::make_shared<std::vector<int>>(*this->shape);
        this->gradient = Tensor::ones(*shapeCopy);
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
        else if (creation_op == "expand_to"){
            this->creator_a->gradient = this->gradient->reduce_to(this->creator_a); 
            this->creator_a->backward();
        }
        else if (creation_op == "reduce_to"){

        }
        else if (creation_op == "sum"){
            this->creator_a->gradient = this->gradient->expand_to(this->creator_a);
            this->creator_a->backward();

        }
    }
}

void Tensor::print() {
    if (dims == 2) {
        int rows = shape->at(0);
        int cols = shape->at(1);
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

std::shared_ptr<Tensor> Tensor::zeros(std::vector<int> shape) {
    int dims = shape.size();

    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Use shared_ptr<float[]> to manage the float array
    std::shared_ptr<float[]> arr(new float[size]());

    auto newShape = std::make_shared<std::vector<int>>(shape);

    // Use make_shared for Tensor to ensure exception safety
    return std::make_shared<Tensor>(arr, newShape);
}

std::shared_ptr<Tensor> Tensor::ones(std::vector<int> shape) {
    int dims = shape.size();

    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Use shared_ptr<float[]> to manage the float array
    std::shared_ptr<float[]> arr(new float[size]);
    for (int i = 0; i < size; i++){
        arr[i] = 1.0f;
    }

    auto newShape = std::make_shared<std::vector<int>>(shape);

    // Use make_shared for Tensor to ensure exception safety
    return std::make_shared<Tensor>(arr, newShape);
}

std::shared_ptr<Tensor> Tensor::randoms(std::vector<int> shape, float min, float max) {
    int dims = shape.size();

    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    // Use shared_ptr<float[]> to manage the float array
    std::shared_ptr<float[]> arr(new float[size]);

    // Setup random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    auto newShape = std::make_shared<std::vector<int>>(shape);

    // Use make_shared for Tensor to ensure exception safety
    return std::make_shared<Tensor>(arr, newShape);
}
