import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

imageH = 96
imageW = 96

transform_train = transforms.Compose([transforms.Resize([imageH, imageW]),   #creates 96x96 image
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    ])


transform_validate = transforms.Compose([transforms.Resize([imageH, imageW]),   #creates 96x96 image
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),   #converts the image to a Tensor
                                    ])

Train_dir="D:/Swati/Thesis Project/Code/Swati_Code/car_data/train"
Validation_dir="D:/Swati/Thesis Project/Code/Swati_Code/car_data/validate"

Train_data = datasets.ImageFolder(Train_dir,       
                    transform=transform_train)

Validation_data = datasets.ImageFolder(Validation_dir,
                   transform=transform_validate)

batch_size = 50
train_load = torch.utils.data.DataLoader(dataset = Train_data, 
                                         batch_size = batch_size,
                                         shuffle = True)

validate_load = torch.utils.data.DataLoader(dataset = Validation_data,
                                        batch_size = batch_size,
                                       shuffle = True)

epoch = 1000
learning_rate = 0.001

class NetworkB(nn.Module):
    def __init__(self):
        super(NetworkB, self).__init__()
        self.layer_B1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride = 2))
        self.layer_B2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3, stride = 2))
        self.layer_B3 = nn.Sequential(
            nn.Linear(24 * 24 * 8, 1024),
            nn.Dropout(),
            nn.Linear(1024, 2),
            nn.Dropout(),
            nn.Softmax(1))
        
    def forward(self, y):
        out_B = self.layer_B1(y)
        out_B = self.layer_B2(out_B)
        out_B = out_B.reshape(out_B.size(0), -1)
        out_B = self.layer_B3(out_B)
        return out_B

print('Building the model...')

NetB = NetworkB()
print(NetB)

#Loss function NetB
criterion_netB = nn.CrossEntropyLoss()
optimizer_netB = optim.SGD(NetB.parameters(), lr=learning_rate, momentum = 0.9)

tr_acc_list = []
train_loss = []
validation_loss = []
vl_acc_list = []
mean = 0
std = 1
                
numSamplesTrain=len(Train_data)
numSamplesVal=len(Validation_data)

print("\nNumber Train Samples", numSamplesTrain)
print("Number Validate Samples", numSamplesVal)

print('\nTraining...')

for ep in range(epoch):
    total_loss1 = 0.0
    correct = 0
    iterations = 0
    val_correct = 0
    val_iterations = 0
    val_loss = 0.0
    
    NetB.train()
    for i, (images, labels) in enumerate(train_load):
        # Regularization (StandardScalar)
        m = images.mean(0, keepdim=True)
        s = images.std(0, unbiased=False, keepdim=True)
        images -= m
        images /= s
        optimizer_netB.zero_grad()           #Clears old gradients from last step
        
        predictions = NetB(images)
        
        NetB_loss = criterion_netB(predictions, labels)
        NetB_loss.backward()
        optimizer_netB.step()
        
        total_loss1 += NetB_loss.item()
        
        # Tracking the train accuracy
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum()
        iterations += 1
    train_loss.append(total_loss1/iterations)
    tr_acc_list.append((100 * correct // numSamplesTrain))
    
    NetB.eval()
    for j, (val_images, val_labels) in enumerate(validate_load):
        # Regularization (StandardScalar)
        v_m = val_images.mean(0, keepdim=True)
        v_s = val_images.std(0, unbiased=False, keepdim=True)
        val_images -= v_m
        val_images /= v_s
        outputs = NetB(val_images)
        Loss = criterion_netB(outputs, val_labels)
        val_loss += Loss.item()
        _, val_predicted = torch.max(outputs, 1)
        val_correct += (val_predicted == val_labels).sum()
        val_iterations += 1
    validation_loss.append(val_loss/val_iterations)
    vl_acc_list.append((100 * val_correct // numSamplesVal))
    
    print('\nEpoch :',ep+1, 'Train_Loss :', train_loss[ep], 'Train_Accuracy :',tr_acc_list[ep], 'Validation_Loss', validation_loss[ep], 'Validation_Accuracy', vl_acc_list[ep])
print('\nFinished Training')

plt.figure(figsize=(10,10))
plt.plot(range(epoch), train_loss, label = 'Train Loss')
plt.plot(range(epoch), validation_loss, label = 'Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show

plt.figure(figsize=(5,5))
plt.plot(range(epoch), tr_acc_list, label = 'Train Accuracy')
plt.plot(range(epoch), vl_acc_list, label = 'Validation Accuracy')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epochs')
plt.legend()
plt.show