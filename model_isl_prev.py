import torch.nn as nn

class ISL_CNN(nn.Module):
  def __init__(self,num_labels):
    super().__init__()
    self.convolution=nn.Sequential(

        # Block 1:
        nn.Conv2d(3,32,3,padding=1), # (3,128,128) -> (32,128,128)
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2), # this divides img dim by 2 so (32,64,64)

         # Block 2:
        nn.Conv2d(32,64,3,padding=1),  # (32,64,64) -> (64,64,64)
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2), # (64,32,32)

         # Block 3:
        nn.Conv2d(64,128,3,padding=1), # (64,32,32) -> (128,32,32) # DO NOT WRITE THAT 1 WITHOUT THE PADDING
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2,2) # (128,16,16)
    )

    self.fully_connected=nn.Sequential(
        nn.Flatten(), # (128,16,16) -> (32768)
        nn.Linear(128*16*16,512), # (512)
        nn.ReLU(),
        nn.Dropout(0.25), # prevents overfitting
        nn.Linear(512,128), # (128)
        nn.ReLU(),
        nn.Linear(128,num_labels) # (nm_labels) which is 40 in this case
    )

  def forward(self,x):
    x=self.convolution(x)
    x=self.fully_connected(x)
    return x

