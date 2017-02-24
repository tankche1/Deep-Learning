import re
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim

currentPath=os.getcwd()
#torch.setdefaulttensortype("torch.FloatTensor")

'''fileName=currentPath+'/person_trainval.txt'
File=open(fileName,'r')
Text=File.read()
#print(Text)

Want=re.findall(r"\d{4,4}\_\d{6,6}\s{2,2}\d",Text)
print(Want)
S=""
for tex in Want:
    S=S+tex[0:11]+'\n'
print(S)
File.close()

fileName=currentPath+'/person_bh.txt'
File=open(fileName,'w+')
File.write(S)
'''

onCuda=1
if(onCuda):
    torch.cuda.manual_seed(1)


fileName=currentPath+'/person_bh.txt'
File=open(fileName,'r')
bh_txt=File.read()
bh_txt=re.findall(r'\d{4,4}\_\d{6,6}',bh_txt)
image_num=len(bh_txt)
print("image_number:%d \n" %(image_num))

validpart={'lhand':0,'rhand':1,'lfoot':2,'rfoot':3,'head':4,'luleg':5,'ruleg':6}
partNum=7
#annot={'lhand':-1,'rhand':-1,'lfoot':-1,'rfoot':-1,'head':-1,'luleg':-1,'ruleg':-1}
anno=torch.Tensor(7,2)

def calc_pos(A):
    x=0;y=0;cnt=0;
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(A[i][j]==1):
                x=x+i;y=y+j;cnt=cnt+1;
    return x/cnt,y/cnt


def getImage(img_bh):
    imgFilename=currentPath+'/Images/'+bh_txt[img_bh]+'.jpg'
    Picture=Image.open(imgFilename)
    #img = np.array(Picture)
    #print(img.shape)
    #plt.imshow(img)
    dst = Picture.resize((60, 60))
    #plt.imshow(dst)
    img=np.array(dst)
    print(img.shape)
    return img

#chM=200 # feature map channels
#chN=100
#chAns=256 # media channels
#ch1=256 # stage channels
#ch2=256

chM=20 # feature map channels
chN=10
chAns=24 # media channels
ch1=24 # stage channels
ch2=24


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, chM/2, 3)  # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3,chM/2,3,padding=1)
        self.conv2 = nn.Conv2d(chM/2, chM, 3,padding=1)

        self.fc1 = nn.Linear(60*60*chN, 100*chN)  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(100*chN, 100)
        self.fc3 = nn.Linear(100,14)


    def Residual(self,x,M,N):
        c=x
        x=F.relu(nn.Conv2d(M,N/2,1)(x))
        x=F.relu(nn.Conv2d(N/2,N/2,1)(x))
        x=F.relu(nn.Conv2d(N/2,N,1)(x))
        c=F.relu(nn.Conv2d(M,N,1)(c))
        x=x+c
        return x

    def Block(self,x,M,N):
        x = self.Residual(x, M, chAns)
        x = self.Residual(x, chAns, chAns)
        x = self.Residual(x, chAns, N)
        return x

    def Down(self,x, M, N):
        x = F.max_pool2d(x, (2, 2))
        #print(x.data)
        x = self.Block(x, M, N)
        return x


    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))

        y1=x
        x=self.Down(y1,chM,ch1)
        y2=x
        x=self.Down(y2,ch1,ch2)

        x=self.Residual(x,ch2,chN)

        x=self.Residual(x,chN,chN)
        x=torch.nn.UpsamplingNearest2d(scale_factor=2)(x)
        y2=self.Block(y2,chAns,chN)
        x=x+y2

        x=self.Residual(x,chN,chN)
        x = torch.nn.UpsamplingNearest2d(scale_factor=2)(x)
        y1=self.Block(y1,chM,chN)
        x = x + y1

        x=x.view(-1,self.num_flat_features(x))
        #print("Watch\nWWWW\n\n\n")
        #print(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net=Net()
if(onCuda):
    net.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.5)
criterion=nn.MSELoss()

'''def Calc(output,Target):
    res=0
    for i in range(0,partNum):
        if((Target[i][0]>0.1) or (Target[i][1] >0.1)):
            res=res+(output[i][0]-Target[i][0])*(output[i][0]-Target[i][0])
            res=res+(output[i][1]-Target[i][1])*(output[i][1]-Target[i][1])
    return res
'''

def Train(img_bh):
    img_name=bh_txt[img_bh]
    fileName=currentPath+'/Annotations_Part/'+img_name+'.mat'
    data=sio.loadmat(fileName)
    print("%s\n" %data['anno'][0][0]['imname'])
    #print(data['anno'][0][0]['objects'].shape)
    objectNum=data['anno'][0][0]['objects'][0].shape[0]
    print("objectNum: %d \n" %objectNum)
    for objectBh in range(0,objectNum):
        if(data['anno'][0][0]['objects'][0][objectBh]['class']==u'person'):
            partsNum=data['anno'][0][0]['objects'][0][objectBh]['parts'].shape[1];
            print('partsNum: %d\n',partsNum)
            #annot = {'lhand': -1, 'rhand': -1, 'lfoot': -1, 'rfoot': -1, 'head': -1, 'luleg': -1, 'ruleg': -1}
            annot=torch.zeros(7,2)
            for partsBh in range(0,partsNum):
                C=str(data['anno'][0][0]['objects'][0][objectBh]['parts'][0][partsBh]['part_name'])
                C=C[3:-2]
                if(C in validpart):
                    partPos=calc_pos(data['anno'][0][0]['objects'][0][objectBh]['parts'][0][partsBh]['mask'])
                    annot[validpart[C]][0]=partPos[0]
                    annot[validpart[C]][1]=partPos[1]
            print(annot)

            # input.reshape(60,180)
            #C=torch.from_numpy(getImage(img_bh).transpose(2,0,1).reshape(1,3,60,60)).float()
            #print(C)
            #input=Variable(torch.from_numpy(getImage(img_bh).transpose(2,0,1).reshape(1,3,60,60)))
            #input = Variable(torch.randn(1, 3, 32, 32))
            input=torch.from_numpy(getImage(img_bh).transpose(2,0,1).reshape(1,3,60,60)).float()
            #print(input.data)
            Target = annot
            if(onCuda):
                Target=Target.cuda()
                input=input.cuda()
            input=Variable(input)
            Target=Variable(Target)
            output=net(input)

            #print(output)


            #output.resize_(7,2)
            print(output)
            print('conv1.bias.grad before backward')
            print(net.conv1.bias.grad)
            optimizer.zero_grad()
            #print(output.data)
            #print(Target.data)
            #Target=Variable(torch.Tensor(7,2))
            #Target=Variable(torch.range(1,14))
            loss=criterion(output,Target)
            print("Loss: ")
            print(loss)
            loss.backward()
            #optimizer.step()
            print('conv1.bias.grad after backward')
            print(net.conv1.bias.grad)


Train(0)
Train_epoch=1
for epoch in range(0,Train_epoch):
    print("Training epoch: %d \n" %epoch)
    for img_bh in range(0,image_num):
        Train(img_bh)
