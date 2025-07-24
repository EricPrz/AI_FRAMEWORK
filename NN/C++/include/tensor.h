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
    std::shared_ptr<std::vector<int>> shape;
    bool requires_grad;
    std::string creation_op;
    int creation_op_arg;
    std::shared_ptr<Tensor> creator_a;
    std::shared_ptr<Tensor> creator_b;
    int size;
    int dims;
    
    Tensor(std::shared_ptr<float[]> dataPtr, std::shared_ptr<std::vector<int>> shape, bool requires_grad = true);
    ~Tensor();

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> sum(int axis, bool keepdims);
    std::shared_ptr<Tensor> expand_to(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> reduce_to(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator-();
    std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> transpose();
    std::shared_ptr<Tensor> power(std::shared_ptr<float> power);
    std::shared_ptr<Tensor> softmax();
    std::shared_ptr<Tensor> softmax(int axis);

    
    void set_creator(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, const std::string& op, int arg = 0);
    void backward();
    
    void print();
    
    
    static std::shared_ptr<Tensor> zeros(std::vector<int> shape);
    static std::shared_ptr<Tensor> ones(std::vector<int> shape);
    static std::shared_ptr<Tensor> randoms(std::vector<int> shape, float min = -1.0f, float max = 1.0f);
};
