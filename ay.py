
import numpy as np
import time
import framework as fm

np.random.seed(1)

inpt = fm.Tensor(np.arange(9).reshape((1, 1, 3, 3)))


class Model(fm.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = fm.Conv2d(1, 1, 2, 1, bias=False)

    def forward(self, x):
        init = time.time()
        x = self.maxpool1.forward(x)
        print("Maxpool forward Time:", time.time() - init)
        return x


model = Model()
model.maxpool1.weights.data = np.ones_like(model.maxpool1.weights.data)
loss_fn = fm.MSE()


pred = model.forward(inpt)

loss = loss_fn.forward(pred, fm.Tensor(np.ones_like(pred.data)))

loss_fn.backward()
