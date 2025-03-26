First of all we need to create a 'Tensor', an object to store our logic and our scalar data.

It will have:
- Data, numpy vector or array, the scalar data.
- Creators, the other Tensors that are direct parents of the current Tensor.
- Creation_op, the method of creation of the Tensor, mathematically wise.
- Autograd, if we want to autograd the Tensor.

Each tensor will be able to do some mathematic operations like summation, multiplications, matrix multiplications, transpose...,
to get an string representation...
