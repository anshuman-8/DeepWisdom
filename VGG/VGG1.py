import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as Dataset
import torchvision.transforms as Transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCH = 20
NUM_CLASSES = 47
BATCH_SIZE = 64
LEARNING_RATE = 0.01

VGG16 = [64, 64, 'M', 128,128,'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']

class VGGNet1(nn.Module):
    def __init__(self, in_channels=1, num_classes=1000, architecture=VGG1):
        super(VGGNet1, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG1)
        self.fcs = nn.Sequential(
            nn.Linear(2304, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048,num_classes)
        )
        self.softMax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fcs(out)
        out = self.softMax(out)
        return out

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)
        

def main():
    #Download the dataset
    train_dataset = Dataset.EMNIST(root="./data", train=True, transform=Transforms.ToTensor(), split="balanced", download=True)
    test_dataset = Dataset.EMNIST(root="./data", train=False, transform=Transforms.ToTensor(), split="balanced", download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mean = [0.1751]
    std = [0.3267]
    num_samples=112800

    normalize_transform = Transforms.Normalize(mean=mean, std=std)

    train_loader.transform = Transforms.Compose([
        Transforms.ToTensor(),
        normalize_transform
    ])

    test_loader.transform = Transforms.Compose([
        Transforms.ToTensor(),
        normalize_transform
    ])

    model = VGGNet1(architecture=VGG1,in_channels=1, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=0.001)

    train_losses, train_acc = train(model,NUM_EPOCH,criterion,optimizer,scheduler, train_loader=train_loader, test_loader=test_loader )

def train(model, epochs, criterion, optimizer, scheduler, train_loader, test_loader):
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    for i in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        best_acc = 0.0

        print(f"Epoch: {i + 1}")

        for images, targets in tqdm(train_loader, desc= "Train\t"):
            images, targets = images.to(device), targets.to(device)

            # images = images.reshape(images.shape[0], -1) 

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            running_correct += (pred == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()

        train_losses.append(running_loss / len(train_loader))
        train_acc.append(running_correct / total)

        running_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, targets in tqdm(test_loader, desc= "Test\t"):
                images, targets = images.to(device), targets.to(device)

                output = model(images)
                preds = torch.argmax(output, dim=1)

                correct += (preds == targets).sum().item()
                running_test_loss += criterion(output, targets).item()
                total += targets.size(0)

            acc = correct / total
            test_acc.append(acc)
            test_losses.append(running_test_loss / len(test_loader))

        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            torch.save(model.state_dict(), './models/model.pth') 

        print(f"Train Loss: {train_losses[-1]:.3f}, Train Acc: {train_acc[-1]:.3f}, Test Loss: {test_losses[-1]:.3f}, Test Acc: {test_acc[-1]:.3f}\n")

    return train_losses, train_acc
            