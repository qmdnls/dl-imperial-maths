import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import inputs and targets
df = pd.read_csv('poverty.txt', sep='\t')

inputs = df.values[:,[1,2]].astype('float32')
targets = df.values[:,4].astype('float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets).unsqueeze(1)

# Create dataset and define data loader
batch_size = 5
train = TensorDataset(inputs, targets)
train_dl = DataLoader(train, batch_size, shuffle=True)

# Define model and optimizer
model = torch.nn.Linear(2, 1)
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
