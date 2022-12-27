import torch.nn as nn
class Model(nn.Module):

    def __init__(self,input_dim,device):
        super(Model, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,39),
            nn.ReLU()
        )
        self.loss=nn.CrossEntropyLoss()
        self.loss.to(device)
    def forward(self,x):
        x=self.model(x)
        x=x.squeeze(1)
        return x
    def cal_loss(self,outputs,targets):
        loss=self.loss.forward(outputs,targets)
        return loss