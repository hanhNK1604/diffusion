import torch 
from torch import nn 
from DownsampleBlock import DownsampleBlock
from UpsampleBlock import UpsampleBlock

class UnetDiffusion(nn.Module):
  def __init__(
      self,
      beta_start=1e-4,
      beta_end=1e-1,
      time_steps=100,
      d_model=64,
      down=[64, 128, 256, 512, 1024],
      up=[1024, 512, 256, 128, 64]
  ):
    super(UnetDiffusion, self).__init__()
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.time_steps = time_steps
    self.d_model = d_model
    self.down = down
    self.up = up

    self.list_beta = torch.linspace(start=beta_start, end=beta_end, steps=time_steps)
    self.list_alpha = 1. - self.list_beta
    self.cumprod_alpha = torch.cumprod(self.list_alpha, dim=0)
    self.sqrt_cumprod_alpha = torch.sqrt(self.cumprod_alpha)
    self.sqrt_one_cumprod_alpha = torch.sqrt(1. - self.cumprod_alpha)


    self.time_embedding = nn.Embedding(time_steps, d_model)

    self.conv2d_input = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    self.encoder = nn.ModuleList([
        DownsampleBlock(in_ch=down[i], out_ch=down[i+1], d_model=d_model) for i in range(len(down)-1)
    ])

    self.bottle_neck = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

    self.decoder = nn.ModuleList([
        UpsampleBlock(in_ch=up[i], out_ch=up[i+1], d_model=d_model) for i in range(len(up) - 1)
    ])

    self.conv2d_output = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(1),
        nn.ReLU(inplace=True)
    )

  def forward_process(self, x: torch.Tensor, list_t: torch.Tensor):
    """
    x: origin image: (bs, c, h, w)
    list_t: list time step (bs,)

    return: list image at time steps: (bs, c, h, w), noise added: (bs, c, h, w)
    """
    list_sqrt_cumprod_alpha = self.sqrt_cumprod_alpha[list_t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    list_sqrt_one_cumprod_alpha = self.sqrt_one_cumprod_alpha[list_t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    noise = torch.rand_like(x)

    out = x*list_sqrt_cumprod_alpha + noise*list_sqrt_one_cumprod_alpha

    return out.to(x.device), noise.to(x.device)

  def forward(self, x: torch.Tensor, t: torch.Tensor):
    """
    x: image added noise: (bs, c, h, w) = (bs, 1, 32, 32)
    list_t: list time step (bs,)

    return: predict_noise: (bs, c, h, w)
    """

    time_embedding = self.time_embedding(t).float() #(bs, d_model)

    input = self.conv2d_input(x) #(bs, 64, 32, 32)

    encoder_list = []
    encoder_list.append(input)
    for net in self.encoder:
      input = net(input, time_embedding)
      encoder_list.append(input)
    
    midle = encoder_list[-1] 
    out = self.bottle_neck(midle)

    for net in self.decoder: 
      encoder_list = encoder_list[:-1]
      out = net(out, encoder_list[-1], time_embedding)
      
    out = self.conv2d_output(out)

    return out 
  
net = UnetDiffusion()

t = torch.tensor(
    [1, 2]
)

x = torch.rand(size=(2, 1, 32, 32))

print(net(x, t).shape)