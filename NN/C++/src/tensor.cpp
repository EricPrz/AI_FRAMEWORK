// tensor.cpp
#include "tensor.h"
#include <random>
#include <cstring> // for memcpy
#include <iostream>
#include <cmath>

Tensor::Tensor(float* dataPtr, int shape[], int dims, bool requires_grad) {
    this->size = 1;
    this->dims = dims;
    this->shape = new int[dims];
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
    delete[] data;
    delete[] shape;
}

Tensor* Tensor::matmul(Tensor* other) {
    int m = this->shape[0];
    int n = this->shape[1];
    int p = other->shape[1];

    float* result = new float[m * p]();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i*p + j] += this->data[i*n + k] * other->data[k*p + j];
            }
        }
    }

    int newShape[2] = {m, p};
    Tensor* tns = new Tensor(result, newShape, 2, this->requires_grad || other->requires_grad);
    tns->set_creator(this, other, "matmul");
    return tns;
}

Tensor* Tensor::operator+(Tensor* other) {
    float* result = new float[this->size];
    for (int i = 0; i < this->size; i++) {
        result[i] = this->data[i] + other->data[i];
    }

    Tensor* tns = new Tensor(result, this->shape, this->dims, this->requires_grad || other->requires_grad);
    tns->set_creator(this, other, "add");
    return tns;
}

Tensor* Tensor::operator-() {
    float* result = new float[this->size];
    for (int i = 0; i < this->size; i++) {
        result[i] = - this->data[i];
    }

    Tensor* tns = new Tensor(result, this->shape, this->dims, this->requires_grad);
    tns->set_creator(this, nullptr, "substract");
    return tns;
}

Tensor* Tensor::operator-(Tensor* other) {
    float* result = new float[this->size];
    for (int i = 0; i < this->size; i++) {
        result[i] = this->data[i] - other->data[i];
    }

    Tensor* tns = new Tensor(result, this->shape, this->dims, this->requires_grad || other->requires_grad);
    tns->set_creator(this, other, "substract");
    return tns;
}

Tensor* Tensor::operator*(Tensor* other) {
    if (this->shape == other->shape){
        float* result = new float[this->size];
        for (int i = 0; i < this->size; i++) {
            result[i] = this->data[i] * other->data[i];
        }

        Tensor* tns = new Tensor(result, this->shape, this->dims, this->requires_grad || other->requires_grad);
        tns->set_creator(this, other, "mul");
        return tns;
    } 
    else if (this->size == 1){
        float* result = new float[other->size];
        for (int i = 0; i < other->size; i++) {
            result[i] = this->data[0] * other->data[i];
        }

        Tensor* tns = new Tensor(result, other->shape, other->dims, this->requires_grad || other->requires_grad);
        tns->set_creator(this, other, "mul");
        return tns;
    }
    else {
        return nullptr;
    }
}

Tensor* Tensor::transpose() {
    float* result = new float[this->size];
    int rows = this->shape[0];
    int cols = this->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = this->data[i * cols + j];
        }
    }

    int newShape[2] = {this->shape[1], this->shape[0]};
    Tensor* tns = new Tensor(result, newShape, 2);
    return tns;
}

Tensor* Tensor::power(float* power){
    float* result = new float[this->size];
    int rows = this->shape[0];
    int cols = this->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = std::pow(this->data[i * cols + j], *power);
        }
    }

    Tensor* tns = new Tensor(result, this->shape, this->dims);

    int shp[] = {1};
    Tensor* powerTns = new Tensor(power, shp, 1);
    tns->set_creator(this, powerTns, "power");
    return tns;
}

Tensor* Tensor::softmax(){
    float *result = new float[this->size];
    float total = 0.0f;

    for (int i = 0; i < this->size; i++){
        result[i] = std::exp(this->data[i]);
        total += result[i];
    }

    for (int i = 0; i < this->size; i++){
        result[i] = result[i]/total;
    }

    Tensor *tns = new Tensor(result, this->shape, this->dims);
    tns->set_creator(this, nullptr, "softmax");

    return tns;
}

void Tensor::set_creator(Tensor* a, Tensor* b, std::string op) {
    this->creator_a = a;
    this->creator_b = b;
    this->creation_op = op;
}

void Tensor::backward() {
    if (!this->requires_grad) return;

    if (this->gradient == nullptr) {
        int* shapeCopy = new int[this->dims];
        std::memcpy(shapeCopy, this->shape, this->dims * sizeof(int));
        this->gradient = Tensor::ones(shapeCopy, this->dims);
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
            Tensor* a_T = this->creator_a;
            Tensor* b_T = this->creator_b;
            Tensor* grad = this->gradient;

            Tensor* grad_a = grad->matmul(b_T->transpose());
            Tensor* grad_b = a_T->transpose()->matmul(grad);

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
    }
}

void Tensor::print() {
    for (int i = 0; i < this->size; i++) {
        std::cout << this->data[i] << " ";
    }
    std::cout << std::endl;
}

Tensor* Tensor::zeros(int shape[], int dims) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    float* arr = new float[size]();
    return new Tensor(arr, shape, dims);
}

Tensor* Tensor::ones(int shape[], int dims) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    float* arr = new float[size];
    for (int i = 0; i < size; i++) arr[i] = 1.0f;
    return new Tensor(arr, shape, dims);
}

Tensor* Tensor::randoms(int shape[], int dims, float min, float max) {
    int size = 1;
    for (int i = 0; i < dims; i++) size *= shape[i];

    float* arr = new float[size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    return new Tensor(arr, shape, dims);
}
