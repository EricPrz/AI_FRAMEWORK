// tensor.h
#pragma once

#include <iostream>
#include <vector>
#include <string>

class Tensor {
  public:
    float *data;
    int id;
    Tensor *gradient;
    int *shape;
    int dims;
    int size;
    bool requires_grad;
    std::string creation_op;
    Tensor *creator_a;
    Tensor *creator_b;
    Tensor(float *dataPtr, int shape[], int dims, bool requires_grad = true);
    ~Tensor();

    Tensor *matmul(Tensor *other);
    Tensor *operator+(Tensor *other);
    Tensor *operator-();
    Tensor *operator-(Tensor *other);
    Tensor *operator*(Tensor *other);
    Tensor *transpose();
    Tensor* power(float power);
    
    void set_creator(Tensor *a, Tensor *b, std::string op);
    void backward();
    
    void print();
    
    
    static Tensor* zeros(int shape[], int dims);
    static Tensor* ones(int shape[], int dims);
    static Tensor* randoms(int shape[], int dims, float min = -1.0f, float max = 1.0f);
};
