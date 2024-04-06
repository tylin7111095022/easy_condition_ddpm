"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple, Optional
from tqdm import tqdm
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from datasets import AICUPDataset


def set_lr(optim, init_lr:float,last_lr:float, total_iter:int,cur_iter:int):
    assert init_lr >= last_lr, "init_lr is greater than last_lr."
    lr_f = lambda init_lr, last_lr, total_iter,i : last_lr if i >= total_iter else init_lr-((i/total_iter)*(init_lr-last_lr))
    cur_lr = lr_f(init_lr, last_lr, total_iter,cur_iter) 
    for g in optim.param_groups:
        g['lr'] = cur_lr

    return optim

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_dim)
        return self.block(x)
    
class Dconv(nn.Module):
    def __init__(self, inc, outc, downsampling:bool=False):
        super(Dconv,self).__init__()
        self.inc = inc
        self.outc = outc
        self.downsampling = downsampling
        self.final_stride = 2 if downsampling else 1
        self.fmap = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outc, outc, 3, self.final_stride,padding=1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
        )
    def forward(self, x):
        self.fmap = self.conv2(self.conv1(x))
        return self.fmap
    
class Uconv(nn.Module):
    def __init__(self, inc, outc):
        super(Uconv,self).__init__()
        self.inc = inc
        self.outc = outc
        self.up = nn.ConvTranspose2d(inc, outc,2,2)
        self.double_conv = Dconv(inc = 2*outc,outc=outc)

    def forward(self, x, enc_f):
        x1 = self.up(x)
        diff_y = enc_f.size(2) - x1.size(2)
        diff_x = enc_f.size(3) - x1.size(3)
        pad_seq = (diff_x//2, diff_x-diff_x//2, diff_y//2, diff_y-diff_y//2)
        x1 = F.pad(x1,pad_seq,mode="constant",value=0)

        f = torch.concat([x1,enc_f],dim=1)
        
        return self.double_conv(f)

class SimpleUnet(nn.Module):
    
    def __init__(self, in_channel: int, channels:list=[48,96,192,384], is_condition:bool=False) -> None:
        """in channel is equal to channel of condition channel concat channel of image channel"""
        super(SimpleUnet, self).__init__()
        self.in_channel = in_channel
        self.everychannels = channels
        self.is_condition = is_condition
        self.everychannels.insert(0,in_channel)
        self.encoder = nn.ModuleList([Dconv(inc=self.everychannels[i],outc=self.everychannels[i+1],downsampling=True) for i in range(len(self.everychannels)-1)])
        self.hiddenconv = Dconv(inc=self.everychannels[-1],outc=self.everychannels[-1],downsampling=False)
        self.decoder = nn.ModuleList([Uconv(inc=self.everychannels[i],outc=self.everychannels[i-1]) for i in range(len(self.everychannels)-1,0,-1)])
        self.out_channel = in_channel//2 if self.is_condition else in_channel
        self.out_conv = nn.Conv2d(channels[0],self.out_channel,1,1,0)  # because in_channel equal to orig_img and condtion image concat at channel axis.
        self.t_embeddings = nn.ModuleList([EmbedFC(input_dim=1,emb_dim=c) for c in self.everychannels])

    def forward(self, x, t, condition) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.

        t_embeddings = [embed(t)[:,:, None, None] for embed in self.t_embeddings]
        if condition is not None:
            x1 = torch.concat([x,condition], dim=1) + t_embeddings[0]
        else: 
            x1 = x + t_embeddings[0]

        h = x1
        for i, e in enumerate(self.encoder):
            h = e(h) + t_embeddings[i+1]
        h = self.hiddenconv(h)
        # decoder
        h = self.decoder[0](h,self.encoder[-2].fmap) + t_embeddings[-2]
        h = self.decoder[1](h,self.encoder[-3].fmap)
        h = self.decoder[2](h,self.encoder[-4].fmap)
        noise = self.decoder[3](h,x1)

        noise = self.out_conv(noise)

        return noise
    
class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        blk = lambda ic, oc: nn.Sequential(
        nn.Conv2d(ic, oc, 7, padding=3),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t, condition:torch.Tensor= None) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor, condition:torch.Tensor= None) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # print(eps.shape)
        # print(self.eps_model(x_t, _ts / self.n_T, condition=condition).shape)

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T, condition=condition))

    def sample(self, n_sample: int,size, condition:torch.Tensor=None, device:str="cuda:0") -> torch.Tensor:
        """生成乾淨的圖 推論過程"""
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        if condition is not None:
            assert condition.shape[0] == n_sample, "batch size dismatch"
            assert condition.shape[1] == size[0] and condition.shape[2] == size[1] and condition.shape[3] == size[2], f"dimension dismatch, condition:{condition.shape}"

        # print(f"x_i shape: {x_i.shape}")

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            t = i / self.n_T
            if isinstance(t,float):
                t = torch.tensor([t]).to(device)
            
            eps = self.eps_model(x_i, t, condition)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        # print(x_i.min(), x_i.max())
        # x_i = x_i.clamp(0.0, 1.0) # 需要限制x_i 的值介於0~1嗎?
        return x_i


def train_aicup(n_epoch: int = 1000, batch:int=50, sample_num:int=4, device="cuda:0",load_pth: Optional[str] = None) -> None:
    if not os.path.exists("./contents"):
        os.makedirs("./contents")
    ddpm = DDPM(eps_model=SimpleUnet(6,is_condition=True), betas=(1e-4, 0.02), n_T=1000)

    if load_pth:
        ddpm.load_state_dict(torch.load(load_pth))
        print("load weight at %s." % load_pth)

    ddpm.to(device)

    dataset = AICUPDataset(data_root=r"data\Training_dataset\img",condition_root=r"data\Training_dataset\label_img")
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0,drop_last=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    total_iters = ((len(dataset) // batch)+1)*n_epoch
    i_iter = 0
    for i in range(n_epoch):
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x_gt, x_cond in pbar: #  in pbar
            set_lr(optim, init_lr=2e-4,last_lr=2e-6,total_iter=total_iters//2,cur_iter=i_iter)
            x_gt = x_gt.to(device)
            x_cond = x_cond.to(device)
            optim.zero_grad()
            loss = ddpm(x_gt, x_cond)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch: {i+1}, i_iter: {i_iter+1}/{total_iters}, loss: {loss_ema:.4f}")
            optim.step()
            i_iter += 1

        ddpm.eval()
        with torch.no_grad():
            cond_img = x_cond[:sample_num]
            gt = x_gt[:sample_num]
            xh = ddpm.sample(sample_num, (3, 120, 214),cond_img, device)
            xset = torch.concat([gt,cond_img,xh],dim=0)
            grid = make_grid(xset, nrow=sample_num)
            save_image(grid, f"./contents/ddpm_sample_{i+1}.png")
            
            # save model
            if i % 50 == 0:
                torch.save(ddpm.state_dict(), f"./contents/ddpm_aicup_{i+1}.pth")
    torch.save(ddpm.state_dict(), f"./contents/ddpm_aicup_last.pth")

    return

def train_cifar10(n_epoch: int = 100, batch:int=100, sample_num:int=4, device="cuda:0") -> None:
    if not os.path.exists("./cifar10"):
        os.makedirs("./cifar10")
    ddpm = DDPM(eps_model=DummyEpsModel(3), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    total_iters = ((len(dataset) // batch)+1)*n_epoch
    i_iter = 0
    for i in range(n_epoch):
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x_gt, _ in pbar:
            set_lr(optim, init_lr=2e-4,last_lr=2e-6,total_iter=total_iters//2,cur_iter=i_iter)
            x_gt = x_gt.to(device)
            optim.zero_grad()
            loss = ddpm(x_gt)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch: {i+1}, i_iter: {i_iter+1}/{total_iters}, loss: {loss_ema:.4f}")
            optim.step()
            i_iter += 1

        ddpm.eval()
        with torch.no_grad():
            gt = x_gt[:sample_num]
            xh = ddpm.sample(sample_num, (3, 32, 32), device=device)
            xset = torch.concat([gt,xh],dim=0)
            grid = make_grid(xset, nrow=sample_num)
            save_image(grid, f"./cifar10/cifar10_{i}.png")
            
            # save model
            if i % 10 == 0:
                torch.save(ddpm.state_dict(), f"./cifar10/cifar10_{i+1}.pth")

    return

if __name__ == "__main__":
    train_aicup(n_epoch=1200,batch=50,sample_num=4,load_pth=r"")
