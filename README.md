#Hourglass

I implement a simpler module of Stacked Hourglass using pytorch.
The code contain the process of loading data from .mat and the whole training parts.
Because I don't have a powerful machine I just reduce the scale of the model to test it on my computer.

Following codes developed the basic risidual model.(x denotes the input data ,M denotes the input channels,N denotes the ouput channels)

  def Residual(self,x,M,N):
 
        c=x
        x=F.relu(nn.Conv2d(M,N/2,1)(x))
        x=F.relu(nn.Conv2d(N/2,N/2,1)(x))
        x=F.relu(nn.Conv2d(N/2,N,1)(x))
        c=F.relu(nn.Conv2d(M,N,1)(c))
        x=x+c
        return x
        
        
The whole module 
  
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
        
        

        
        
        
        
     
