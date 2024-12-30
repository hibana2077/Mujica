# from unet_model import UNet_Var, UNet, VovUnet_Var
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from medmnist.dataset import OrganAMNIST, OrganCMNIST
import os
import json
import argparse

# define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# check './data' directory
if not os.path.exists('./data'):
    os.makedirs('./data')

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define dataset
train_dataset = OrganCMNIST(root='./data', split='train', transform=transform, download=True, size=224)
test_dataset = OrganCMNIST(root='./data', split='test', transform=transform, download=True, size=224)

# define dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# check dataset size
print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

for i, (img, label) in enumerate(train_loader):
    print(f"Batch {i} - Image shape: {img.shape}, Label shape: {label.shape}")
    break

# define model
mo
model = model.to(device)

# define loss function
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = nn.MSELoss()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
LR_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# define training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.squeeze(1)
        optimizer.zero_grad()
        outputs_seg, outputs_cls = model(inputs)
        loss_cls = criterion_cls(outputs_cls, targets)
        loss_seg = criterion_seg(outputs_seg, inputs)
        loss = loss_cls + loss_seg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs_cls.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(f"[Train] Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}",end='\r')
    
    print(f"[Train] Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}")
    return train_loss/total, 100. * correct / total

# define testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze(1)
            outputs_seg, outputs_cls = model(inputs)
            loss_cls = criterion_cls(outputs_cls, targets)
            loss_seg = criterion_seg(outputs_seg, inputs)
            loss = loss_cls + loss_seg
            test_loss += loss.item()
            _, predicted = outputs_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {test_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}",end='\r')

    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {test_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}")
    return test_loss/total, 100. * correct/total

# define main function
def main():
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(10):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        LR_scheduler.step()
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

    # save training history
    history = {
        'train_loss': train_loss_hist,
        'train_acc': train_acc_hist,
        'test_loss': test_loss_hist,
        'test_acc': test_acc_hist
    }
    with open('history.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()