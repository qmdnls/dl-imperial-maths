import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import inputs and targets
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Create dataset and define data loader
batch_size = 5
train = TensorDataset(inputs, targets)
train_dl = DataLoader(train, batch_size, shuffle=True)

# Define model and optimizer
model = torch.nn.Linear(3, 2)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Define our loss function
loss_compute = F.mse_loss

# This is our training function
def fit(num_epochs, model, loss_compute, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Generate predictions and compute loss
            pred = model(xb)
            loss = loss_compute(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_compute(model(inputs), targets))

# Let's train the model for 100 epochs
fit(100, model, loss_compute, opt)

preds = model(inputs)
print(preds)
print(targets)
