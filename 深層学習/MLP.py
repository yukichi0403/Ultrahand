import torch
from torch import nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self,num_in, num_hidden, num_out):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(num_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        z1 = self.l1(self.flatten(x))
        a1 = F.relu(z1)
        z2 = self.l2(a1)
        return z2

class MyDataset(Dataset):
    def __init__(self, X, y ,transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)

        return X, y

def learn(model, num_epochs, train_loader, val_loader, optimizer, loss_func, early_stopping_rounds=None):
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    no_improve_rounds = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        running_val_acc = 0.0

        for train_batch, data in enumerate(train_loader):
            X,y = data
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_func(preds,y)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()


        with torch.no_grad():
            for val_vatch, data in enumerate(val_loader):
                X_val, y_val = data
                preds_val = model(X_val)
                loss = loss_func(preds_val,y_val)
                running_val_loss += loss.item()

                val_acc = torch.sum(torch.argmax(preds_val,dim=-1) == y_val) / y_val.shape[0]
                running_val_acc += val_acc.item()

        train_losses.append(running_loss / (train_batch + 1))
        val_losses.append(running_val_loss / (val_batch + 1))
        val_accuracies.append(running_val_acc / (val_batch + 1))
    
        if val_losses[-1] < best_val_loss:
            no_improve_rounds = 0
            best_val_loss = val_losses[-1]
        else:
            no_improve_rounds += 1
    
        
        if early_stopping_rounds and no_improve_rounds >= early_stopping_rounds:
            print("stopping_early")
            break

        print(f"epoch: {epoch + 1}: train error: {train_losses[-1]},validation error: {val_losses[-1]}, validation accuracy: {val_accuracies[-1]}")

    return train_losses, val_losses, val_accuracies



#使用例
dataset = datasets.load_digits()
images = dataset['images']
images = images * (255./16.)
images = images.astype(np.uint8)

target = dataset['target']

X_train, X_val, y_train, y_val = train_test_split(images, target, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

batch_size = 32

train_dataset = MyDataset(X_train,y_train,transform = transform)
val_dataset = MyDataset(X_val,y_val,transform=transform)

train_loader = DataLoader(train_dataset, batch_size = batch_size,shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size = batch_size , num_workers=2)

learning_rate = 0.1
model = MLP(8*8, 30, 10)
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)

train_losses, val_losses, val_accuracies = learn(model,50,train_loader, val_loader, optimizer=opt, loss_func=F.cross_entropy, early_stopping_rounds=None)
