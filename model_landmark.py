import torch.nn as nn

class ISL_Landmark(nn.Module):
  def __init__(self,num_labels):
    super().__init__()
    self.convolution=nn.Sequential(

        # Block 1:
        nn.Linear(127,256), # (50859, 127) -> (50859, 256)
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),

         # Block 2:
        nn.Linear(256,128), # (50859, 256) -> (50859, 128)
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128,64), # (50859, 128) -> (50859, 64)
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Linear(64,len(class_names))

    )

    # self.fully_connected=nn.Sequential(
    #     nn.Flatten(), # (128,16,16) -> (32768)
    #     nn.Linear(128*16*16,512), # (512)
    #     nn.ReLU(),
    #     nn.Dropout(0.25), # prevents overfitting
    #     nn.Linear(512,128), # (128)
    #     nn.ReLU(),
    #     nn.Linear(128,num_labels) # (nm_labels) which is 40 in this case
    # )

  def forward(self,x):
    x=self.convolution(x)
    # x=self.fully_connected(x)
    return x