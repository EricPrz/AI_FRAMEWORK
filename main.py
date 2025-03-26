import framework as fm
import pandas as pd
import numpy as np
import time
import pickle

train = pd.read_csv("./data/mnist_train.csv")
test = pd.read_csv("./data/mnist_test.csv")

# Reestructuring
train_label, train_image = train["label"].to_numpy(), train[[x for x in train.columns if x != "label"]].to_numpy().reshape((-1, 1, 28, 28))/255
test_label, test_image = train["label"].to_numpy(), train[[x for x in train.columns if x != "label"]].to_numpy().reshape((-1, 1, 28, 28))/255

# One-hotting
one_hot = np.zeros((len(train_label), 10))
for label in range(train_label.shape[0]):
    one_hot[label, train_label[label]] = 1
train_label = one_hot

one_hot = np.zeros((len(test_label), 10))
for label in range(test_label.shape[0]):
    one_hot[label, test_label[label]] = 1
test_label = one_hot

del train
del test
del one_hot

np.random.seed(1)

class Model(fm.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = fm.Conv2d(1, 6, 5, 1, bias = False)
        self.relu1 = fm.ReLu()
        self.maxpool1 = fm.MaxPool2d(2, 6, 2)
        self.conv2 = fm.Conv2d(6, 16, 5, 1, bias = False)
        self.maxpool2 = fm.MaxPool2d(2, 16, 2)
        self.lin1 = fm.Linear(256, 84)
        self.lin2 = fm.Linear(84, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.maxpool1.forward(x)
        x = self.conv2.forward(x) 
        x = self.maxpool2.forward(x)
        x = x.reshape((-1, 256))
        x = self.lin1.forward(x) 
        x = self.lin2.forward(x)
        return x


model = Model()

# Load the object back from the file
#with open('prueba.pkl', 'rb') as file:
#    model = pickle.load(file)

#print("Object loaded successfully.")



loss_fn = fm.CrossEntropy()
#optim = fm.Adadelta(model.get_parameters())
optim = fm.Adam(model.get_parameters())


batch_size = 128
num_batches = int(len(train_label)/batch_size)
epochs = 4

init = time.time()
for epoch in range(epochs):

    print(f"Epoch {epoch+1}\n-------------------------------")

    total_loss = 0
    correct = 0

    for batch in range(num_batches):
        batch_start, batch_end = (batch * batch_size, (batch + 1) * batch_size)

        inpt = fm.Tensor(train_image[batch_start:batch_end])
        label = fm.Tensor(train_label[batch_start:batch_end])
        
        pred = model.forward(inpt)

        pred_nums = np.argmax(pred.data, axis=1)
        label_nums = np.argmax(label.data, axis=1)

        correct += (pred_nums == label_nums).sum()
        
        loss = loss_fn.forward(pred, label)

        total_loss += loss  

        if batch % int(num_batches/7) == 0:
            print(f"loss: {loss} [{(batch+1)*batch_size}/{batch_size*num_batches}]")   
        
        loss_fn.backward()
        
        optim.step()
        

    print("Avg error: ", total_loss / num_batches) 
    print("Batch accuracy:", correct / (num_batches * batch_size) * 100, "%\n") 

print("Train time:", time.time() - init)




# Testing
correct = 0
for batch in range(num_batches):
        batch_start, batch_end = (batch * batch_size, (batch + 1) * batch_size)

        inpt = fm.Tensor(test_image[batch_start:batch_end])
        label = fm.Tensor(test_label[batch_start:batch_end])
        
        pred = model.forward(inpt)

        pred_nums = np.argmax(pred.data, axis=1)
        label_nums = np.argmax(label.data, axis=1)

        correct += (pred_nums == label_nums).sum()

print("Accuracy:", correct / (batch_size * num_batches) * 100, "%")

model.save("prueba")
