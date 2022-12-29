import torch.nn as nn
class Model(nn.Module):

    def __init__(self,input_dim,device):
        super(Model, self).__init__()
        self.model=nn.Sequential(
            # nn.Linear(input_dim,1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(1024,1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(1024,512),
            # nn.ReLU(),
            # nn.Linear(512,39)

            nn.Linear(input_dim,1024),
            nn.Linear(1024,512),
            nn.Linear(512,128),
            nn.Linear(128,39),
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