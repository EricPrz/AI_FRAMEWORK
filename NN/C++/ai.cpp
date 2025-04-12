#include <iostream>
#include <random>

class Tensor {
  public:
    float *data;
    int id;
    float *gradient;
    int *shape;
    int dims;
    int size;
    Tensor(float *dataPtr, int shape[], int dims){
      data = dataPtr;
      gradient = NULL;
      id = 0;
      this->size = 1;
      this->shape = new int[dims];
      for (int i = 0; i < dims; i++){
        this->size *= shape[i];
        this->shape[i] = shape[i];
      }
      this->dims = dims;
    }
    Tensor *matmul(Tensor *other){
      float *result = new float[this->shape[0] * other->shape[1]]();
      for(int j = 0; j < this->shape[0]; j++){
        for(int i = 0; i < other->shape[1]; i++){
          for(int x = 0; x < this->shape[1]; x++){
            result[j * other->shape[1] + i] += this->data[j * this->shape[1] + x] * other->data[x * other->shape[1] + i];
          }
        }
      }
      int shape[2] = {this->shape[0], other->shape[1]};
      Tensor *tns = new Tensor(result, shape, 2);
      return tns;
    }
    Tensor *operator+(Tensor *other){
      float *result = new float[this->size];
      if(this->size == other->size * other->size){
        for (int j = 0; j < this->shape[0]; j++){
          for (int i = 0; i < this->shape[1]; i++){
            int flatCoord = j * this->shape[1] + i;
            result[flatCoord] = this->data[flatCoord] + other->data[i];
          }
        }
      } else{
        for (int j = 0; j < this->shape[0]; j++){
          for (int i = 0; i < this->shape[1]; i++){
            int flatCoord = j * this->shape[1] + i;
            result[flatCoord] = this->data[flatCoord] + other->data[flatCoord];
          }
        }
      }
      Tensor *tns = new Tensor(result, this->shape, this->dims);
      return tns;
    }
    void print(){
      for(int i = 0; i < this->size; i++){
        std::cout << this->data[i] << " ";
      }
    }
};

Tensor *zeros(int shape[], int dims){
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= shape[i];
  }

  float *arr;
  arr = new float[size];

  for(int i = 0; i < size; i++){
    arr[i] = 0;
  }

  Tensor *tns = new Tensor(arr, shape, dims);
  return tns;
}

Tensor *ones(int shape[], int dims){
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= shape[i];
  }

  float *arr;
  arr = new float[size];

  for(int i = 0; i < size; i++){
    arr[i] = 1;
  }

  Tensor *tns = new Tensor(arr, shape, dims);
  return tns;
}

Tensor *randoms(int shape[], int dims, float min = -1.0, float max = 1.0){
    int size = 1;
    for (int i = 0; i < dims; i++) {
        size *= shape[i];
    }

    float *arr = new float[size];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(min, max);

    for (int i = 0; i < size; i++) {
        arr[i] = distr(gen);
    }

    return new Tensor(arr, shape, dims);
}


class Layer {
  public:
    Tensor *parameters;
    Layer(){
      parameters = NULL;
    }
};

class Linear: public Layer{
  public:
    bool hasBias;
    Tensor *bias;
    Tensor *weights;
    Linear(int in_features, int out_features, bool bias = true){
      if (bias){
        int shape[] = {1, out_features};
        this->bias = zeros(shape, 2);
      }
      this->hasBias = bias;
      int shape[2] = {in_features, out_features};
      this->weights = randoms(shape, 2);
    }
    Tensor *forward(Tensor *inpt){
      Tensor *mul = inpt->matmul(this->weights);
      if (this->bias){
        return *mul + this->bias;
      } else{
        return mul;
      }
    }
};


int main() {
  int shape[] = {64, 128};
  Tensor *inpt = randoms(shape, 2);
  Linear *lin1 = new Linear(128, 64);
  Tensor *res = lin1->forward(inpt);
  res->print();
  return 0;
}
