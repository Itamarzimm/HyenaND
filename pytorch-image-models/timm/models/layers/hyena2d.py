"""
Simplified standalone version of Hyena: https://arxiv.org/abs/2302.10866, designed for quick experimentation.
A complete version is available under `src.models.sequence.hyena`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

def augment(v):
    v1 =  v.flip(-1,-2)
    v2 =  v.flip(-1)
    v3 =  v.flip(-2)
    t = torch.cat((v,v1,v2,v3), dim=0)
    return t       

def is_square(n):
    root = int(math.sqrt(n))
    return root*root == n

def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

def fftconv2d(u, k, D):
    dtype_out = u.dtype
    Yseqlen = u.shape[-1]
    Xseqlen = u.shape[-2]
    assert Yseqlen == Xseqlen
    seqlen = Xseqlen
    fft_size = 2 * seqlen
    #k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = torch.fft.rfft2(k.float(), s=(fft_size, fft_size)) / fft_size
    
    #u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    u_f = torch.fft.rfft2(u.float(), s=(fft_size, fft_size))
    
    #if len(u.shape) > 3: k_f = k_f.unsqueeze(1)


    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    y = torch.fft.irfft2(u_f * k_f, s=(fft_size, fft_size)) [..., :seqlen,:seqlen]
    out = y + u * D.unsqueeze(1).unsqueeze(2)
    return out.to(dtype=dtype_out)

def fftconv2dcp(u, k, D):
    dtype_out = u.dtype
    Yseqlen = u.shape[-1]
    Xseqlen = u.shape[-2]
    assert Yseqlen == Xseqlen
    seqlen = Xseqlen
    fft_size = 2 * seqlen

    def fft_operation(u, k):
        k_f = torch.fft.rfft2(k.float(), s=(fft_size, fft_size)) / fft_size
        u_f = torch.fft.rfft2(u.float(), s=(fft_size, fft_size))
        y = torch.fft.irfft2(u_f * k_f, s=(fft_size, fft_size), norm='forward')[..., :seqlen, :seqlen]
        return y

    out = checkpoint(fft_operation, u, k)
    out = out + u * D.unsqueeze(1).unsqueeze(2)

    return out.to(dtype=dtype_out)


@torch.jit.script 
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)
            

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)
    
    
class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs): 
        """Complex exponential positional embeddings for Hyena filters."""  
        super().__init__()
        
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2            
        # To compute the right embeddings we use the "proper" linspace 
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1 
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None] 
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb) 
        self.register("t", t, lr=0.0)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]
    

class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        window_mode = "standard", #["learn", "symettric", "no" ,"standard"]
        modulate: bool=True,
        shift: float = 0.0,
        hyena2d_filter = False,
        sqrt_decay = False,
        **kwargs
    ):
        super().__init__()
        self.window_mode = window_mode
        self.sqrt_decay = sqrt_decay
        self.hyena2d_filter = hyena2d_filter
        self.modulate = modulate
        if self.window_mode == "no":
            self.modulate = False
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        if not self.hyena2d_filter:
            deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
            if self.window_mode == "standard": # "standard", #["learn", "symettric", "standard"]
                self.register("deltas", deltas, lr=modulation_lr)
            elif self.window_mode == "learn":
                self.deltas = torch.nn.Parameter(deltas)
                self.shift = torch.nn.Parameter(torch.rand(deltas.shape))
            elif self.window_mode == "symettric":
                raise NotImplementedError("no suppurted in product mode")
        else: 
            Xdeltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
            Ydeltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
            if self.window_mode == "standard": 
                self.register("Xdeltas", Xdeltas, lr=modulation_lr)
                self.register("Ydeltas", Ydeltas, lr=modulation_lr)
            elif self.window_mode == "learn":
                self.Xdeltas = torch.nn.Parameter(Xdeltas)
                self.Ydeltas = torch.nn.Parameter(Ydeltas)
                self.shift = torch.nn.Parameter(torch.rand(Xdeltas.shape))
                self.shift = torch.nn.Parameter(torch.rand(Ydeltas.shape))
            elif self.window_mode == "symettric":
                self.deltas = torch.nn.Parameter(Xdeltas)
                self.shift = torch.nn.Parameter(torch.rand(Xdeltas.shape))
            
        print("Create Hyena2D ExponentialModulation with: sqrt-{} and 2d-filter-{}, window_mode-{}".format(sqrt_decay,hyena2d_filter,window_mode))
    def forward(self, t, x):
        if self.modulate:
            if not self.hyena2d_filter:
                decay = torch.exp(-t * self.deltas.abs()) 
                x = x * (decay + self.shift)
            else:
                if self.window_mode == "symettric":
                    power_indices = torch.arange(x.shape[-2]).cuda()
                    powers = (power_indices.unsqueeze(0) + power_indices.unsqueeze(1)).unsqueeze(0).unsqueeze(-1)
                    decay = torch.exp((-1) * powers * self.deltas.abs())
                else:
                    Xdecay = torch.exp(-t * self.Xdeltas.abs()) 
                    Ydecay = torch.exp(-t * self.Ydeltas.abs())
                    decay = torch.einsum('ijk,imk->ijmk', Xdecay, Ydecay)
       
                if self.sqrt_decay:
                    decay = decay**0.5
                x = x * (decay + self.shift)
        return x                  


class HyenaFilter2d(OptimModule):
    def __init__(
            self, 
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP 
            fused_fft_conv=False,
            seq_len=1024, 
            lr=1e-3, 
            lr_pos_emb=1e-5,
            dropout=0.0, 
            w=1, # frequency of periodic activations 
            wd=0, # weight decay of kernel parameters 
            bias=True,
            window_mode = "standard", #["learn", "symettric", "no" ,"standard"]
            num_inner_mlps=2,
            normalized=False,
            hyena2d_filter=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.hyena2d_filter = hyena2d_filter
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        
        #x axes
        if not self.hyena2d_filter:
            self.Yimplicit_filter = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.Yimplicit_filter.append(nn.Linear(order, order))
                self.Yimplicit_filter.append(act)

            self.Yimplicit_filter.append(nn.Linear(order, d_model, bias=False))
                
            self.Ymodulation = ExponentialModulation(d_model, window_mode=window_mode, **kwargs)

            #y axes
            self.Ximplicit_filter = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.Ximplicit_filter.append(nn.Linear(order, order))
                self.Ximplicit_filter.append(act)

            self.Ximplicit_filter.append(nn.Linear(order, d_model, bias=False))
                
            self.Xmodulation = ExponentialModulation(d_model,  window_mode=window_mode, **kwargs)
        
            self.normalized = normalized
            for c in self.Ximplicit_filter.children():
                for name, v in c.state_dict().items():        
                    optim = {"weight_decay": wd, "lr": lr}
                    setattr(getattr(c, name), "_optim", optim)
            for c in self.Yimplicit_filter.children():
                for name, v in c.state_dict().items():        
                    optim = {"weight_decay": wd, "lr": lr}
                    setattr(getattr(c, name), "_optim", optim)
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim*2, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)

            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
                
            self.modulation = ExponentialModulation(d_model,  window_mode=window_mode, hyena2d_filter=self.hyena2d_filter, **kwargs)
            
            self.normalized = normalized
            for c in self.implicit_filter.children():
                for name, v in c.state_dict().items():        
                    optim = {"weight_decay": wd, "lr": lr}
                    setattr(getattr(c, name), "_optim", optim)

    def filter2d(self, L, *args, **kwargs):
        if not self.hyena2d_filter:
            z, t = self.pos_emb(L)
            hx = self.Ximplicit_filter(z)
            hx = self.Xmodulation(t, hx)
            hy = self.Yimplicit_filter(z)
            hy = self.Ymodulation(t, hy)
            h = torch.einsum('ijk,imk->ijmk', hx, hy)
        else:
            z, t = self.pos_emb(L)
            z = z.repeat(1,L,1,1)
            z = torch.cat((z.transpose(-2,-3) , z), dim = -1) 
            h = self.implicit_filter(z) #h shape: [1,L,L,C]
            h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 
        # grad_cp = False
        # if grad_cp:
        #     y = fftconv2dcp(x, k, bias)    
        # else:
        y = fftconv2d(x, k, bias)
        return y
    
    
class HyenaOperator2d(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            hyena2d_filter = False,
            direct_paramtrization = False,
            window_mode = "standard", #["learn", "symettric", "no" ,"standard"]
            directional_mode = "standard", #["standard", "seq", "parallel" ,"4dir","4dirp"]
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        print("Create Hyena2D layer with dirct paramtirzation:{},directional mode:{}, 2dfilter:{}. window_mode={}".format(direct_paramtrization,directional_mode,hyena2d_filter,window_mode))
        self.direct_paramtrization = direct_paramtrization
        self.directional_mode = directional_mode
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        conv_out_chan = inner_width if directional_mode != "parallel" else inner_width*4
        if self.directional_mode == "4dir":
            self.linear_mix = nn.Linear(4, 1)
        if self.directional_mode == "4dirp":
            self.linear_mix = nn.Linear(4*d_model, d_model)

            
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model if directional_mode != "parallel" else d_model*4, d_model)
        self.short_filter = nn.Conv2d( # define conv2d from same size to same size
            inner_width, 
            conv_out_chan, 
            3,
            padding=1,
            groups=inner_width
        )

        if not self.direct_paramtrization:
            self.filter_fn = HyenaFilter2d(
                d_model * (order - 1), 
                order=filter_order, 
                seq_len=l_max,
                channels=1, 
                dropout=filter_dropout, 
                hyena2d_filter = hyena2d_filter,
                window_mode = window_mode,
                **filter_args
            ) 
        else:
            self.filter = torch.nn.Parameter(torch.randn(order, d_model, l_max, l_max))
            self.bias = torch.nn.Parameter(torch.randn(order,d_model))
            
    
    def forward(self, u, *args, **kwargs): # B, H,W,C        
        reshape = False
        B , l = u.shape[0], u.shape[-2]
        if len(u.shape) == 3:
            if is_square(l):
                reshape = True
                u =  rearrange(u, 'b (h w) c -> b h w c', h=int(math.sqrt(l)))
            else:
                print("unput to hyena2d:", u.shape)
                raise ValueError("not square matrix hyena2d")
        #u # B,H,W,C
        l1 = u.size(-2)
        l2 = u.size(-3)
        assert l1 == l2 == self.l_max
        l = l1
        l_filter = min(l1, l2, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l1 l2 d -> b d l1 l2')
        uc = self.short_filter(u)[...,:l_filter, :l_filter] 
        #assert u.shape == uc.shape
        if self.directional_mode == "parallel":
            uc = rearrange(uc, 'b (h c) l1 l2 -> (b h) c l1 l2', h=4)
        *x, v = uc.split(self.d_model, dim=1)
        assert (x)[0].shape == v.shape
        if not self.direct_paramtrization:
            k = self.filter_fn.filter2d(l_filter)[0]
            k = rearrange(k, 'l1 l2 (o d) -> o d l1 l2', o=self.order - 1)
            bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)
        else:
            k = self.filter
            bias = self.bias
        
 
        if self.directional_mode in ["4dir",'4dirp']:
            v = augment(v)
            
        for o, x_i in enumerate(reversed(x[1:])):
            if self.directional_mode in ["4dir",'4dirp']:
                x_i = augment(x_i)

            v = self.dropout(v * x_i)
            
            if self.directional_mode == "seq" and self.order > 1 and o == 0:
                v = v.flip(-1,-2)
            if not self.direct_paramtrization:
                v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])
            else:
                v = fftconv2d(v, k[o], bias[o])

            if self.directional_mode == "seq" and self.order > 1 and o == 0:
                v = v.flip(-1,-2)
        if self.directional_mode in ["4dir",'4dirp']:
            x[0] = augment(x[0])
        y = rearrange(v * x[0], 'b d l1 l2 -> b l1 l2 d')

        if self.directional_mode in ["4dir",'4dirp']:
            y1 =  y[:B,:,:,:]
            y2 = y[B:(2*B),:,:,:].flip(-2,-3)
            y3 = y[(2*B):(3*B),:,:,:].flip(-2)
            y4 = y[(3*B):,:,:,:].flip(-3)
            y = torch.stack([y1,y2,y3,y4],dim=-1) # B, L1,L2,C,4
            if self.directional_mode == "4dir":
                y = self.linear_mix(y).squeeze(-1)
            else:
                y = rearrange(y, 'b l1 l2 c h -> b l1 l2 (c h)')
                y = self.linear_mix(y)
        if self.directional_mode == "parallel":
            y = rearrange(y, '(b h) l1 l2 d -> b l1 l2 (h d)', h = 4)
        y = self.out_proj(y)

        if reshape:
            y =  rearrange(y, 'b h w c -> b (h w) c')
        return y

    
    
if __name__ == "__main__":
    b = 3
    l_max =16
    c = 96
    print('1')
    layer = HyenaOperator2d(
        d_model=c, 
        l_max=l_max, 
        order=2, 
        filter_order=8
    )
    print('2')
    layer1 = HyenaOperator2d(
        d_model=c, 
        l_max=l_max, 
        order=2, 
        filter_order=8,
        directional_mode="seq"
    ) 
    print('3')
    layer2 = HyenaOperator2d(
        d_model=c, 
        l_max=l_max, 
        order=2, 
        filter_order=8,
        directional_mode="parallel"
    )
    print('4')
    layer3 = HyenaOperator2d(
        d_model=c, 
        l_max=l_max, 
        order=2, 
        filter_order=8,
        directional_mode="4dir"
    )
    print('5')
    layer4 = HyenaOperator2d(
        d_model=c, 
        l_max=l_max, 
        order=2, 
        filter_order=8,
        direct_paramtrization = True,
    )

    x = torch.randn(b, l_max,l_max, c, requires_grad=True)
    y = layer(x)   
    print(x.shape, y.shape)
    y = layer1(x)   
    print(x.shape, y.shape)
    y = layer2(x)   
    print(x.shape, y.shape)
    y = layer3(x)   
    print(x.shape, y.shape)
    y = layer4(x)   
    print(x.shape, y.shape)

    print("window mode, product:")
    print()
    for window_mode in ["learn", "no" ,"standard"]:
        print("Wondow mode:", window_mode)
        layer = HyenaOperator2d(
            d_model=c, 
            l_max=l_max, 
            window_mode = window_mode,
            order=2, 
            filter_order=8
        )
        x = torch.randn(b, l_max,l_max, c, requires_grad=True)
        y = layer(x)   
        print(x.shape, y.shape)

    print("window mode, hyena2d_filter:")
    print()
    for window_mode in ["learn", "symettric", "no" ,"standard"]:
        print("Wondow mode:", window_mode)
        layer = HyenaOperator2d(
            d_model=c, 
            l_max=l_max, 
            window_mode = window_mode,
            hyena2d_filter = True,
            order=2, 
            filter_order=8
        )
        x = torch.randn(b, l_max,l_max, c, requires_grad=True)
        y = layer(x)   
        print(x.shape, y.shape)
        
    # grad = torch.autograd.grad(y[:, 10, :].sum(), x)[0]
    # print('Causality check: gradients should not flow "from future to past"')
    # print(grad[0, 11, :].sum(), grad[0, 9, :].sum())
