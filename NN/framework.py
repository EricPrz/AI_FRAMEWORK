import ctypes
import numpy as np
import time
import pickle
import platform

os = platform.system()
# Load the shared library
if os == "Windows":
    lib = ctypes.CDLL("./lib/functions.exe")
elif os == "Linux":
    lib = ctypes.CDLL("./lib/functions.so")

# Ensure the function returns a POINTER to float
lib.getMaxPool2D.restype = ctypes.POINTER(ctypes.c_float)

lib.getSects.restype = ctypes.POINTER(ctypes.c_float)

class Tensor(object):
    def __init__(self, data, creators=None, creation_op=None, autograd=True, is_parameter=False):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.creators = creators
        self.creation_op = creation_op
        self.autograd = autograd
        self.gradient = None
        self.is_parameter = is_parameter
    
    def backward(self, gradient):
        if not (self.autograd or self.creators):
            return
    
        self.gradient = gradient
        
        if self.creation_op == "add":
            self.creators[0].backward(self.gradient)
            self.creators[1].backward(self.gradient)
        elif self.creation_op == "sub":
            self.creators[0].backward(self.gradient)
            self.creators[1].backward(-self.gradient)
        elif self.creation_op == "pow":
            self.creators[0].backward(self.creators[1] * self.gradient)
        elif self.creation_op == "transpose":
            self.creators[0].backward(self.gradient.transpose())
        elif self.creation_op == "dot":
            # Gradient of weights
            self.creators[0].backward(self.gradient.dot(self.creators[1].transpose()))
            # Gradient of input
            self.creators[1].backward(self.gradient.transpose().dot(self.creators[0]).transpose())
        elif self.creation_op == "matmul":
            # Gradient of weights
            self.creators[0].backward(self.gradient.dot(self.creators[1].transpose()))
            # Gradient of input
            self.creators[1].backward(self.gradient.reshape((-1, self.gradient.shape[-1])).dot(self.creators[0].reshape((self.creators[0].shape[0], -1))).transpose())
        elif self.creation_op == "mul":
            self.creators[0].backward(self.gradient * self.creators[1])
            self.creators[1].backward(self.gradient * self.creators[0])
        elif self.creation_op == "div":
            self.creators[0].backward(self.gradient / self.creators[1])
            self.creators[1].backward(self.gradient * self.creators[0] / -(self.creators[1]**2))
        elif self.creation_op == "relu":
            self.creators[0].backward(Tensor(self.gradient.data * (self.creators[0].data > 0)))
        elif self.creation_op == "batchnorm":
            creator = self.creators[0]
            self.creators[1].backward(Tensor(self.gradient.data * creator.gamma.data / np.sqrt(creator.var + creator.epsilon)))
            creator.gamma.backward(Tensor(self.gradient.data * (self.creators[1].data-creator.mean)/np.sqrt(creator.var + creator.epsilon)))
            creator.beta.backward(self.gradient)

        elif self.creation_op == "get_sects":      
            """
            conv = self.creators[0]
            x = self.creators[1]

            new_gradient = np.zeros(x.data.shape)

            for i in range(conv.n ** conv.n):
                print("Idxs c:", conv.pos[:, :, i])
                print("Grad:", self.gradient.data.reshape(conv.pos.shape)[:, :, i])
                print("Idxs:", idxs)
                print("New Grad:", new_gradient[idxs])
                idxs = np.unravel_index(conv.pos[:, :, i], x.data.shape)
                new_gradient[idxs] += self.gradient.data.reshape(conv.pos.shape)[:, :, i]   



            
            x.backward(Tensor(new_gradient))

            """

            init_time = time.time()
            conv = self.creators[0]
            x = self.creators[1]

            new_gradient = np.zeros_like(x.data)
            base_idxs = np.arange(x.data.shape[2] * x.data.shape[3]).reshape((x.data.shape[2], x.data.shape[3]))
           
            for i in range(conv.n ** 2):
                coords = conv.extracted_from_pos[i]
                idxs = base_idxs[coords["y_from"]:coords["y_to"]:coords["step"], coords["x_from"]:coords["x_to"]:coords["step"]].flatten()
               
                row_idxs, col_idxs = np.unravel_index(idxs, (x.shape[2], x.shape[3]))
            
                new_gradient[:, :, row_idxs, col_idxs] += self.gradient.data.reshape((x.shape[0], conv.in_channels, conv.kernel_size * conv.kernel_size, conv.n * conv.n))[:, :, :, i]
            #print("Sects Back Time:", time.time() - init_time)
            x.backward(Tensor(new_gradient))



        elif self.creation_op == "reshape":
            self.creators[0].backward(self.gradient.reshape(self.creators[0].data.shape))
        elif self.creation_op == "get_max2d": 
            init = time.time()
            # Initialize variables and precompute the base index array
            maxpool = self.creators[0]
            x = self.creators[1]
            new_gradient = np.zeros(x.shape)

            idxs = np.unravel_index(maxpool.extracted_from_pos, (x.shape))
            """ print("Maxpool grad time:", time.time() - init) """
            new_gradient[idxs] += self.gradient.data.flatten()
            
            x.backward(Tensor(new_gradient))
 
        del self
            



    def __add__(self, other):
        return Tensor(self.data + other.data, creators=[self, other], creation_op="add")
    
    def relu(self):
        return Tensor(self.data * (self.data > 0), creators=[self], creation_op="relu")
    
    def __sub__(self, other):
        return Tensor(self.data - other.data, creators=[self, other], creation_op="sub")
    
    def __neg__(self):
        return Tensor(self.data * -1, creators=[self], creation_op="neg")
    
    def __mul__(self, other):
        return Tensor(self.data * other.data, creators=[self, other], creation_op="mul")

    def __truediv__(self, other):
        return Tensor(self.data * other.data, creators=[self, other], creation_op="div")
    
    def __pow__(self, exponent):
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        return Tensor(np.power(self.data, exponent.data), creators=[self, exponent], creation_op="pow")

    def dot(self, other):
        return Tensor(np.dot(self.data, other.data), creators=[self, other], creation_op="dot")
    
    def matmul(self, other):
        return Tensor(np.matmul(self.data, other.data), creators=[self, other], creation_op="matmul")

    def transpose(self):
        return Tensor(self.data.T, creation_op="transpose", creators=[self])
    
    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), creation_op="reshape", creators=[self])
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
    
class Module:
    def __init__(self):
        self.parameters = list()
        self.training = False

    def get_super_atributes(self):
        return set(dir(super()))

    def get_atributes(self):
        return set(self.__dict__.keys())

    def get_parameters(self):
        temp = [self.__dict__[x].get_parameters() for x in self.get_atributes() if isinstance(self.__dict__[x], Layer)]
        temp = [x for x in temp if x != []]

        for parameters in temp:
            for pars in parameters:
                self.parameters.append(pars)
        del temp
        return self.parameters

    def get_children(self):    
        return [x for x in self.get_atributes() if isinstance(self.__dict__[x], Layer)]

    def train(self):
        self.training = True   
        for module in self.get_children():
            self.__dict__[module].training = True

    def test(self):
        self.training = False
        for module in self.get_children():
            self.__dict__[module].training = True
 

    def save(self, fileName):
        # Save the object to a file
        with open(fileName+'.pkl', 'wb') as file:
            pickle.dump(self, file)

        print("Object saved successfully.")
    
class Optimizer(object):
    def __init__(self, params):
        self.parameters = params

class Adagrad(Optimizer):
    def __init__(self, params, lr = 0.007, epsilon = 1e-10, autoZero = True):
        super().__init__(params)
        self.lr = lr
        self.epsilon = epsilon
        self.autoZero = autoZero
        
        self.G = [np.zeros(x.data.shape) for x in self.parameters]

    def step(self):
        for j in range(len(self.parameters)):
            param = self.parameters[j]
            
            self.G[j] = self.G[j] + param.gradient.data ** 2
            param.data = param.data - self.lr / np.sqrt(self.G[j] + self.epsilon) * param.gradient.data

            if self.autoZero:
                param.gradient.data *= 0

class Adadelta(Optimizer):
    def __init__(self, params, lr = 1, gamma = 0.9, epsilon = 1e-6, autoZero = True):
        super().__init__(params)
        self.epsilon = epsilon
        self.autoZero = autoZero
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        
        self.vt = [np.zeros(x.data.shape) for x in self.parameters]
        self.ut = [np.zeros(x.data.shape) for x in self.parameters]

    def step(self):
        for j in range(len(self.parameters)):
            param = self.parameters[j]
            
            gt = param.gradient.data

            self.vt[j] = self.vt[j] * self.gamma + gt ** 2 * (1 - self.gamma)

            xt = np.sqrt(self.ut[j] + self.epsilon) / np.sqrt(self.vt[j] + self.epsilon) * gt

            self.ut[j] = self.ut[j] * self.gamma + xt ** 2 * (1 - self.gamma)
            
            param.data = param.data - self.lr * xt

            if self.autoZero:
                param.gradient.data *= 0    

class Adam(Optimizer):
    def __init__(self, params, lr = 0.0025, gamma_m = 0.9, gamma_v = 0.999, epsilon = 1e-6, autoZero = True):
        super().__init__(params)
        self.lr = lr
        self.m_t = [np.zeros(x.data.shape) for x in self.parameters]
        self.v_t = [np.zeros(x.data.shape) for x in self.parameters]
        self.mc_t = [np.zeros(x.data.shape) for x in self.parameters]
        self.vc_t = [np.zeros(x.data.shape) for x in self.parameters]
        self.gamma_m = gamma_m
        self.gamma_v = gamma_v
        self.autoZero = autoZero
        self.epsilon = epsilon

    def step(self):
        for j in range(len(self.parameters)):
            param = self.parameters[j]
            
            self.m_t[j] = self.gamma_m * self.m_t[j] + (1 - self.gamma_m) * param.gradient.data
            self.v_t[j] = self.gamma_v * self.v_t[j] + (1 - self.gamma_v) * param.gradient.data * param.gradient.data

            self.mc_t[j] = self.m_t[j] / (1 - self.gamma_m)
            self.vc_t[j] = self.v_t[j] / (1 - self.gamma_v)

            param.data = param.data - self.lr / (np.sqrt(self.vc_t[j]) + self.epsilon) * self.mc_t[j]

            if self.autoZero:
                param.gradient.data *= 0

class SGD(Optimizer):
    def __init__(self, params, lr = 0.01, momentum = True, gamma = 0.9, autoZero = True):
        super().__init__(params)
        self.lr = lr

        self.autoZero = autoZero
        self.momentum = momentum

        self.gamma = gamma

        if momentum:            
            self.vt = [np.zeros(x.data.shape) for x in self.parameters]


    def step(self):
        for j in range(len(self.parameters)):
            param = self.parameters[j]
            
            if self.momentum:
                vt = self.vt[j]     
                vt = self.gamma * vt + self.lr * param.gradient.data
                param.data = param.data - vt
            else:
                param.data = param.data - self.lr * param.gradient.data 

            if self.autoZero:
                param.gradient.data *= 0



class Layer(object):
    def __init__(self):
        self.parameters = list()
        self.training = False

    def get_parameters(self):
        return self.parameters

class Loss(object):
    def __init__(self):
        self.loss = None

    def backward(self):
        self.loss.backward(Tensor(np.ones(self.loss.data.shape)))

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred:Tensor, y:Tensor):
        self.pred = pred
        """ self.loss = (pred - y) * (pred - y) """
        self.loss2 = ((pred.data - y.data) ** 2).sum(1).mean()
        self.grad = 2 * (pred.data - y.data)
        
        return self.loss2

    def backward(self):
        self.pred.backward(Tensor(self.grad))

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred:Tensor, y:Tensor):
        self.pred = pred

        aux = np.exp(pred.data)
        softmax = aux / np.sum(aux, axis=1, keepdims = True)
        del aux
        loss = -(np.log(softmax) * y.data).sum(1).mean()
        gradient = softmax - y.data
        del softmax
        self.grad = gradient
        return loss

    def backward(self):
        self.pred.backward(Tensor(self.grad))
    


class Linear(Layer):
    def __init__(self, in_channels:int, out_channels:int, bias:bool=True):
        super().__init__()

        xavier_range = np.sqrt(6 / (in_channels + out_channels))
        self.weights = Tensor(np.random.uniform(-xavier_range, xavier_range, size=(in_channels, out_channels)), is_parameter=True)
        self.parameters.append(self.weights)

        self.hasBias = bias

        if bias:
            self.bias = Tensor(np.zeros((1, out_channels)), is_parameter=True)
            self.parameters.append(self.bias)

    def forward(self, x:Tensor):
        mm = x.dot(self.weights)
        if self.hasBias:
            return mm + self.bias
        else:
            return mm

class Dropout(Layer):
    def __init__(self, pct):
        super().__init__()
        self.pct = pct
        self.mask = None

    def forward(self, x:Tensor):
        if not self.training:
            return x

        if not self.mask:
            self.mask = Tensor(np.random.binomial(1, self.pct, x.shape))
        return x * self.mask

class BatchNorm(Layer):
    def __init__(self, epsilon = 1e-5, momentum = 0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None
        
        # Running statistics for inference phase
        self.running_mean = None
        self.running_var = None

    def NormBatched(self, x:Tensor):
        return Tensor((x.data - self.mean)/np.sqrt(self.var + self.epsilon) * self.gamma.data + self.beta.data, creation_op="batchnorm", creators=[self, x])

    def forward(self, x:Tensor):

        if self.gamma is None:
            # Initialize scale (gamma) and shift (beta) for each feature (channel)
            self.gamma = Tensor(np.ones((1, x.shape[1], 1, 1)), is_parameter=True)
            self.beta = Tensor(np.zeros((1, x.shape[1], 1, 1)), is_parameter=True)
            self.parameters.append(self.gamma)
            self.parameters.append(self.beta)
        
        if self.running_mean is None:
            # Initialize running mean and variance to zeros
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])
        
        if self.training:
            self.mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(x.data, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            return self.NormBatched(x)
        else:
            x_reshape = x.reshape((-1, x.shape[1]))
            return ((x_reshape - self.running_mean)/(self.running_var + self.epsilon)**0.5 * self.gamma + self.beta).reshape(x.shape)

class ReLu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor):
        return x.relu()
    
class MaxPool2d(Layer):
    def __init__(self, kernel_size, in_channels, stride=1, padding=0, dilation=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def get_max2d(self, x:Tensor):
        
        n = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        
        if n % 1 != 0:
            raise Exception("No valid pooling parameters")
        
        n = int(n)

        self.n = n
        self.extracted_from_pos = np.zeros(x.shape[0] * self.in_channels * n * n, dtype=np.int32)
        
        extracted_from_pos_ptr = self.extracted_from_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        x_data_ptr = x.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        pooledarr_ptr = lib.getMaxPool2D(
            x.data.shape[0], self.in_channels, x.data.shape[2], x.data.shape[3],
            x_data_ptr,
            self.kernel_size, self.in_channels, self.stride, self.dilation, self.padding, extracted_from_pos_ptr
        )

        pooledarr = np.ctypeslib.as_array(pooledarr_ptr, shape=(x.data.shape[0], self.in_channels, n, n))

        maxTns = Tensor(pooledarr, creators=[self, x], creation_op="get_max2d")
        
        lib.cleanPtr(pooledarr_ptr)

        return maxTns

    def forward(self, x:Tensor):
        return self.get_max2d(x)


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        xavier_range = np.sqrt(6 / (in_channels + out_channels))
        self.weights = Tensor(np.random.uniform(-xavier_range, xavier_range, size=(kernel_size * kernel_size * in_channels, out_channels)), is_parameter=True)
        self.parameters.append(self.weights)

        self.hasBias = bias

        if bias:
            self.bias = Tensor(np.zeros(out_channels), is_parameter=True)
            self.parameters.append(self.bias)

    def get_sects(self, x:Tensor):

        n = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        
        if n % 1 != 0:
            raise Exception("No valid convolution parameters")
        
        n = int(n)

        self.n = n

        sects = np.zeros(shape=(x.data.shape[0], self.in_channels, n * n, self.kernel_size, self.kernel_size))
        self.extracted_from_pos = []

        for i in range(n ** 2):
            x_from = (i%n)*self.stride
            x_to = x_from + self.kernel_size + self.dilation - 1
            y_from = int(i/n) * self.stride
            y_to = y_from + self.kernel_size + self.dilation - 1
            step = self.dilation

            res = x.data[:, :, y_from:y_to:step, x_from:x_to:step]
            
            sects[:, :, i] = res

            self.extracted_from_pos.append({"x_from": x_from, "x_to": x_to, "y_from": y_from, "y_to": y_to, "step": step})


        return Tensor(sects.reshape((x.shape[0] * n * n, self.kernel_size * self.kernel_size * self.in_channels)), creators=[self, x], creation_op="get_sects")
 



    def forward(self, x:Tensor):
        sects = self.get_sects(x)
        mm = sects.dot(self.weights).reshape((x.shape[0], self.out_channels, self.n, self.n))
        
        if self.hasBias:
            return mm + self.bias
        else:
            return mm






class Conv2dC(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        xavier_range = np.sqrt(6 / (in_channels + out_channels))
        self.weights = Tensor(np.random.uniform(-xavier_range, xavier_range, size=(kernel_size * kernel_size * in_channels, out_channels)), is_parameter=True)
        self.parameters.append(self.weights)

        self.hasBias = bias

        if bias:
            self.bias = Tensor(np.zeros(out_channels), is_parameter=True)
            self.parameters.append(self.bias)

    def get_sects(self, x:Tensor):
           
        n = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        if n % 1 != 0:
            raise Exception("No valid convolution parameters")
        n = int(n)
        self.n = n

        self.N = x.shape[0]
        self.chs = x.shape[1]
        self.dims = x.shape[2]

        self.pos = np.zeros(shape=(self.N, self.chs, n * n, self.kernel_size * self.kernel_size), dtype=np.int32)

        sects_ptr = lib.getSects(x.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), self.N, self.chs, self.n, self.dims, self.stride, self.kernel_size, self.dilation)
        
        sects = np.ctypeslib.as_array(sects_ptr, shape=(x.data.shape[0], self.in_channels, self.n, self.n, self.kernel_size, self.kernel_size))

        return Tensor(sects.reshape((x.shape[0] * n * n, self.kernel_size * self.kernel_size * self.in_channels)), creators=[self, x], creation_op="get_sects")

        




    def forward(self, x:Tensor):
        sects = self.get_sects(x)
        mm = sects.dot(self.weights).reshape((x.shape[0], self.out_channels, self.n, self.n))
        
        if self.hasBias:
            return mm + self.bias
        else:
            return mm
