// tensor.h
#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <string>

class Tensor : public std::enable_shared_from_this<Tensor> {
  public:
    std::shared_ptr<float[]> data;
    int id;
    std::shared_ptr<Tensor> gradient;
    std::shared_ptr<int[]> shape;
    int dims;
    int size;
    bool requires_grad;
    std::string creation_op;
    std::shared_ptr<Tensor> creator_a;
    std::shared_ptr<Tensor> creator_b;
    
    Tensor(std::shared_ptr<float[]> dataPtr, int shape[], int dims, bool requires_grad = true);
    ~Tensor();

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator-();
    std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> transpose();
    std::shared_ptr<Tensor> power(std::shared_ptr<float> power);
    std::shared_ptr<Tensor> softmax();
    
    void set_creator(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, const std::string& op);
    void backward();
    
    void print();
    
    
    static std::shared_ptr<Tensor> zeros(int shape[], int dims);
    static std::shared_ptr<Tensor> ones(int shape[], int dims);
    static std::shared_ptr<Tensor> randoms(int shape[], int dims, float min = -1.0f, float max = 1.0f);
};
