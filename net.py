# Taking inspiration from twitch chess https://github.com/geohot/twitchchess
import tinygrad
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
#!/usr/bin/env python3
import numpy as np
import tinygrad.nn as nn
from torch.utils.data import Dataset
from tinygrad.jit import TinyJit
from sklearn.metrics import mean_squared_error as MSE

# from torch.nn.utils import clip_grad_norm_
# from torch import optim

class ChessValueDataset(Dataset):
  def __init__(self):
    dat = np.load("processed/dataset_2Mt.npz")
    self.X = dat['arr_0']
    self.Y = dat['arr_1']
    print("loaded", self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])


class tinyChessNet:
    def __init__(self):
        super(tinyChessNet, self).__init__()
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn_a1 = nn.BatchNorm2d(16)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_a2 = nn.BatchNorm2d(16)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn_a3 = nn.BatchNorm2d(32)
        
        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_b1 = nn.BatchNorm2d(32)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_b2 = nn.BatchNorm2d(32)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn_b3 = nn.BatchNorm2d(64)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.bn_c1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.bn_c2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.bn_c3 = nn.BatchNorm2d(128)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.bn_d1 = nn.BatchNorm2d(128)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)
        self.bn_d3 = nn.BatchNorm2d(128)

        self.last = nn.Linear(128, 1)
        

    def __call__(self, x):
        x = self.a1(x)
        x = self.bn_a1(x)
        x = x.leaky_relu(x)
        # x = self.res_block1(x)
        x = self.a2(x)
        x = self.bn_a2(x)
        x = x.leaky_relu(x)
        x = self.a3(x)
        x = self.bn_a3(x)
        x = x.leaky_relu(x)
        # 4x4
        x = self.b1(x)
        x = self.bn_b1(x)
        x = x.leaky_relu(x)
        # x = self.res_block2(x)
        x = self.b2(x)
        x = self.bn_b2(x)
        x = x.leaky_relu(x)
        x = self.b3(x)
        x = self.bn_b3(x)
        x = x.leaky_relu(x)
        # 2x2
        x = self.c1(x)
        x = self.bn_c1(x)
        x = x.leaky_relu(x)
        # x = self.res_block3(x)
        x = self.c2(x)
        x = self.bn_c2(x)
        x = x.leaky_relu(x)
        x = self.c3(x)
        x = self.bn_c3(x)
        x = x.leaky_relu(x)
        # 1x128
        x = self.d1(x)
        x = self.bn_d1(x)
        x = x.leaky_relu(x)
        # x = self.res_block4(x)
        x = self.d2(x)
        x = self.bn_d2(x)
        x = x.leaky_relu(x)
        x = self.d3(x)
        x = self.bn_d3(x)
        x = x.leaky_relu(x)
        x = x.view(-1, 128)
        x = self.last(x)
        # value output
        return x.tanh(x)
    
if __name__ == "__main__":
  device = "cuda"
  chess_dataset = ChessValueDataset()
  train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)
  model = tinyChessNet()
  optimizer = optim.Adam([model.a1.weight,model.a2.weight,model.a3.weight,model.b1.weight,model.b2.weight,model.b3.weight,model.c1.weight,model.c2.weight,model.c3.weight,model.d1.weight,model.d2.weight,model.d3.weight,model.last.weight], lr = 12e-4)
  floss = MSE
  # if device == "cuda":
    # model.cuda()
  Tensor.training = True
  for epoch in range(100):
    all_loss = 0
    num_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      batch_size = data.size(0)
            # Skip batches with size other than 1024
      if batch_size % 256 != 0:
        continue
      target = target.view(-1, 1)
      data, target = data.to(device), target.to(device)
      data = data.float()
      target = target.float()
      
      # print(data.shape, target.shape)
      optimizer.zero_grad()

      output = model(data)
      
      # to do make this MSE error operate with tinygrad tensors 
      loss = floss(output, target)
      loss.backward()
      # Gradient clipping
      # max_grad_norm = 1.0  # Set the maximum gradient norm value
      # clip_grad_norm_([model.l1.weight, model.l2.weight], max_grad_norm)
      optimizer.step()
         
      all_loss += loss.item()
      num_loss += 1
    print("%3d: %f" % (epoch, all_loss/num_loss))
    torch.save(model.state_dict(), "nets/valueTinyChessNet01.pth")
      
