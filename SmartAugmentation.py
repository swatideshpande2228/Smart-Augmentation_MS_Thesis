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

Train_dir="D:/Swati/Thesis Project/Code/Swati_Code/GenderClassificationDataset/Train"
Validation_dir="D:/Swati/Thesis Project/Code/Swati_Code/GenderClassificationDataset/Validate"

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

    
def save_image(img):
    torchvision.utils.save_image(img,
                              '{}\Aug_Img{}.jpg'.format('/workspace/Dataset', i),
                              normalize=True)

epoch = 1000
alpha = 0.3
netAInpchannels = 3
beta = 0.7
learning_rate = 0.001


class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        self.layer_A1 = nn.Sequential(
            nn.Conv2d(netAInpchannels, 16, kernel_size=3, stride=1, padding=2))
        self.layer_A2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2))
        self.layer_A3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2))
        self.layer_A4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2))
        self.layer_A5 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        out_A = self.layer_A1(x)
        out_A = self.layer_A2(out_A)
        out_A = self.layer_A3(out_A)
        out_A = self.layer_A4(out_A)
        out_A = self.layer_A5(out_A)
        return out_A
    
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
    
NetA = NetworkA()
print(NetA)

NetB = NetworkB()
print(NetB)

#Loss function NetA
criterion_netA = nn.MSELoss()
optimizer_netA = optim.SGD(NetA.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=0.01)

#Loss function NetB
criterion_netB = nn.CrossEntropyLoss()
optimizer_netB = optim.SGD(NetB.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=0.01)

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

def Get_Right_Data(load):
    Images1 = []
    Images2 = []
    inputs, target = load
    # Regularization (StandardScalar)
    m = inputs.mean(0, keepdim=True)
    s = inputs.std(0, unbiased=False, keepdim=True)
    inputs -= m
    inputs /= s
    for tar in target:
        if tar == 0:
            Images1.append(inputs)
            X = random.choices(Images1 , k=netAInpchannels) #Selects 3 Random images from Class0
            X = torch.stack(X, dim=1)
            img0 = X[:,:,0,:]                      #Reshapes tensor [batch_size, netAInpchannels, 96,96]
            img1 = random.choice(Images1)          #Selects random image from Class0
            return img0, img1, target
        if tar == 1:
            Images2.append(inputs)
            X1 = random.choices(Images2 , k=netAInpchannels) #Selects 3 Random images from Class1
            X1 = torch.stack(X1, dim=1)
            img00 = X1[:,:,0,:]                    #Reshapes tensor [batch_size, netAInpchannels, 96,96]
            img11 = random.choice(Images2)         #Selects random image from Class1
            return img00, img11, target
        
            
for ep in range(epoch):
    total_loss1 = 0.0
    correct = 0
    iterations = 0
    val_correct = 0
    val_iterations = 0
    val_loss = 0.0
    
    NetA.train()
    NetB.train()
    for i, data in enumerate(train_load):
        images, img, target = Get_Right_Data(data)
        
        optimizer_netA.zero_grad()             #Clears old gradients from last step
        
        ##Training NetworkA (Augmenter)
        AugImg = NetA(images)                   #Network 1 [Augmenter]
        save_image(AugImg)
        
        InpB = torch.cat([AugImg, img], dim=1)  #Merges output from NetA and random Image from same class
        InpB = InpB[:,0,:,:].unsqueeze(1)
        
        NetA_loss = criterion_netA(AugImg, img)
        
        optimizer_netB.zero_grad()           #Clears old gradients from last step
        
        predictions = NetB(InpB)             #Network 2 [Classifier], Augmented image used for training netB
        
        NetB_loss = criterion_netB(predictions, target)
        NetB_loss.backward(retain_graph=True)
        
        total_loss = NetA_loss * alpha + NetB_loss * beta
        total_loss.backward()               #Backprops total loss [from NetB to NetA]
        
        total_loss1 += total_loss.item()
        
        optimizer_netA.step()                #Updates the weights
        optimizer_netB.step()
        
        # Tracking the train accuracy
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == target).sum()
        iterations += 1
    train_loss.append(total_loss1/iterations)
    tr_acc_list.append((100 * correct // numSamplesTrain))
    
    NetB.eval()
    for j, (val_images, labels) in enumerate(validate_load):
        # Regularization (StandardScalar)
        v_m = val_images.mean(0, keepdim=True)
        v_s = val_images.std(0, unbiased=False, keepdim=True)
        val_images -= v_m
        val_images /= v_s
        outputs = NetB(val_images)
        
        Loss = criterion_netB(outputs, labels)
        val_loss += Loss.item()
        _, val_predicted = torch.max(outputs, 1)
        val_correct += (val_predicted == labels).sum()
        val_iterations += 1
    validation_loss.append(val_loss/val_iterations)
    vl_acc_list.append((100 * val_correct // numSamplesVal))
    
    print('\nEpoch :',ep+1, 'Train_Loss :', train_loss[ep], 'Train_Accuracy :',tr_acc_list[ep], 'Validation_Loss', validation_loss[ep], 'Validation_Accuracy', vl_acc_list[ep])
print('\nFinished Training')

PATH = 'D:/Swati/Thesis Project/Code/Swati_Code/MODEL SAVED/SA_MODEL_Gender.pth'
torch.save(NetB.state_dict(), PATH)

plt.figure(figsize=(5,5))
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