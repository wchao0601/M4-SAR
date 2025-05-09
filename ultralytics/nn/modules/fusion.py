import math
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
# from torchdiffeq import odeint_adjoint as odeint # pip install torchdiffeq
# import pywt # pip install pywavelets
from einops import rearrange, repeat
from scipy import ndimage
from functools import partial
import copy
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
import selective_scan_cuda_core as selective_scan_cuda

# try:
#     from selective_scan import selective_scan_fn as selective_scan_fn_v1
# except:
#     pass


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # input_t: float, fp16, bf16; weight_t: float;
        # u, B, C, delta: input_t
        # D, delta_bias: float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and (D.dtype != torch.float):
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and (delta_bias.dtype != torch.float):
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()
        
        assert u.shape[1] % (B.shape[1] * nrows) == 0 
        assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile
        if u.dtype == 'cpu':
            dtype = 'cpu'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            u = u.to(device)
            delta = delta.to(device)
            A = A.to(device)
            B = B.to(device)
            C = C.to(device)
            D = D.to(device)
            delta_bias = delta_bias.to(device)
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            out = out.to(device)
            x = x.to(device)
        else:
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        
        _dD = None
        if D is not None:
            if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                _dD = dD.to(ctx._d_dtype)
            else:
                _dD = dD

        _ddelta_bias = None
        if delta_bias is not None:
            if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
            else:
                _ddelta_bias = ddelta_bias

        return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)


def selective_scan_fn_v1(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        else:
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    out = out.to(dtype=dtype_in)
    return out


def sparse_mask(in_dim, out_dim):
    '''
    get sparse mask
    '''
    in_coord = torch.arange(in_dim) * 1/in_dim + 1/(2*in_dim)
    out_coord = torch.arange(out_dim) * 1/out_dim + 1/(2*out_dim)

    dist_mat = torch.abs(out_coord[:,None] - in_coord[None,:])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1,0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1,0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:,0], all_connection[:,1]] = 1.
    
    return mask

def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    '''
    #print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    #print('mat', mat.shape)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    #print('y_eval', y_eval.shape)
    device = mat.device
    
    #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    except:
        print('lstsq failed')
    
    # manual psuedo-inverse
    '''lamb=1e-8
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]'''
    
    return coef


def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        
        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            margin = 0.00
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        
        self.grid.data = extend_grid(grid, k_extend=self.k)
        #print('x_pos 2', x_pos.shape)
        #print('y_eval 2', y_eval.shape)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        
        '''
        # based on samples
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid'''
        
        #print('p', parent.grid)
        # based on interpolating parent grid
        def get_grid(num_interval):
            x_pos = parent.grid[:,parent.k:-parent.k]
            #print('x_pos', x_pos)
            sp2 = KANLayer(in_dim=1, out_dim=self.in_dim,k=1,num=x_pos.shape[1]-1,scale_base_mu=0.0, scale_base_sigma=0.0).to(x.device)

            #print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            #print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim), x_pos.permute(1,0).unsqueeze(dim=2), sp2.grid[:,:], k=1).permute(1,0,2)
            shp = sp2_coef.shape
            #sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            #print('sp2_coef',sp2_coef)
            #print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1,1,self.num+1).to(self.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1,0)
            #print('c', grid)
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

class OS_KANFusion1(nn.Module):
    """
    光学与SAR特征融合模块 (基于KAN架构)
    
    输入: 
        optical_feat (Tensor): 光学图像特征 [B, C, H, W]
        sar_feat (Tensor): SAR图像特征 [B, C, H, W]
    输出:
        fused_feat (Tensor): 融合后特征 [B, C, H, W]
    """
    def __init__(self, feat_dim=256, grid_num=5, k=3, grid_eps=0.1, device='cpu'):
        super().__init__()
        
        # ==================== 特征对齐层 ====================
        # 使用1x1卷积统一特征维度(可选，若输入维度不一致时使用)
        self.optical_align = nn.Conv2d(feat_dim, feat_dim, 1)
        self.sar_align = nn.Conv2d(feat_dim, feat_dim, 1)
        
        # ==================== KAN权重生成器 ====================
        # 输入维度: 2*feat_dim (光学+SAR拼接后的通道数)
        # 输出维度: 1 (生成空间自注意力权重)
        self.kan = KANLayer(
            in_dim=2*feat_dim,   # 拼接后的输入维度
            out_dim=1,           # 输出权重维度
            num=grid_num,        # 网格划分数量
            k=k,                 # 样条阶数
            grid_eps=grid_eps,   # 网格采样混合系数
            scale_base_mu=0.1,   # 基函数缩放均值
            scale_base_sigma=0.5, # 基函数缩放方差
            device=device
        )
        
        # ==================== 初始化参数 ====================
        nn.init.kaiming_normal_(self.optical_align.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.sar_align.weight, mode='fan_out')
        self.to(device)

    def forward(self, x):
        """
        前向传播过程:
        1. 对齐特征维度
        2. 拼接特征作为KAN输入
        3. 生成空间自适应融合权重
        4. 加权融合两模态特征
        """
        device=x[0].device
        self.to(device)
        optical_feat = x[0]
        sar_feat =x[0]
        # ===== 1. 特征对齐 =====
        optical_aligned = optical_feat  # [B,C,H,W]
        sar_aligned = sar_feat             # [B,C,H,W]
        
        # ===== 2. 特征拼接 =====
        batch_size, channels, height, width = optical_aligned.shape
        # 沿通道维度拼接光学与SAR特征
        concat_feat = torch.cat([optical_aligned, sar_aligned], dim=1)  # [B,2C,H,W]
        # 转换为KAN输入格式 [B*H*W, 2C]
        kan_input = concat_feat.permute(0,2,3,1).reshape(-1, 2*channels)
        
        # ===== 3. KAN生成权重 =====
        # 通过KAN生成每个空间位置的融合权重
        weights, _, _, _ = self.kan(kan_input)  # weights形状 [B*H*W, 1]
        # 调整权重形状并归一化到[0,1]
        weights = weights.view(batch_size, height, width, 1).permute(0,3,1,2)  # [B,1,H,W]
        weights = torch.softmax(weights,dim=1)  # 使用sigmoid约束权重范围
        
        # ===== 4. 特征融合 =====
        fused_feat = optical_aligned + (1 - weights) * sar_aligned
        return fused_feat

class DynamicRoutingCell(nn.Module):
    """动态路由单元（DeepSeek-R1核心技术）"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 路由条件生成器
        self.condition_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, 3*in_channels, 1),
            nn.Tanh()
        )
        
        # 可变形卷积增强
        # self.deform_conv = DeformConv2d(in_channels, in_channels, 3)
        
    def forward(self, x_opt, x_sar):
        # 生成路由条件
        cond = self.condition_gen(x_opt + x_sar)
        gamma_opt, gamma_sar, deform_offset = cond.chunk(3, dim=1)
        
        # 模态间动态权重
        x_opt_weighted = x_opt * (0.5 + gamma_opt)
        x_sar_weighted = x_sar * (0.5 + gamma_sar)
        # x_opt_weighted = x_opt * gamma_opt
        # x_sar_weighted = x_sar * gamma_sar
        
        # 可变形特征对齐
        # aligned_opt = self.deform_conv(x_opt_weighted, deform_offset)
        # aligned_sar = self.deform_conv(x_sar_weighted, deform_offset)
        # aligned_opt = self.deform_conv(x_opt_weighted)
        # aligned_sar = self.deform_conv(x_sar_weighted)
        
        return x_opt_weighted + x_sar_weighted 

class CrossModalRecombiner(nn.Module):
    """跨模态特征重组器"""
    def __init__(self, in_channels):
        super().__init__()
        # 频域融合
        self.spectral_fusion = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x_opt, x_sar):
        # 频域权重
        spectral_weight = self.spectral_fusion(torch.cat([x_opt, x_sar], dim=1))
        
        # 空间注意力
        # max_pool = torch.max(x_opt, x_sar).mean(dim=1, keepdim=True)
        # avg_pool = torch.mean(torch.stack([x_opt, x_sar]), dim=0)
        # spatial_weight = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))
        
        # 动态重组
        return (x_opt * spectral_weight + x_sar * (1 - spectral_weight)) #* spatial_weight

class OS_KANFusion(nn.Module):
    """DeepSeek-R1 跨模态动态路由主模块"""
    def __init__(self, in_channels):
        super().__init__()
        # 预处理分支
        self.opt_pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )
        self.sar_pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels)
        )
        
        # 动态路由核心
        self.router = DynamicRoutingCell(in_channels)
        
        # 特征重组
        self.recombiner = CrossModalRecombiner(in_channels)
        
        # 残差连接
        self.skip = nn.Conv2d(2*in_channels, in_channels, 1)
        
    def forward(self, x):
        optical = x[0]
        sar = x[1]
        # 预处理
        opt_feat = self.opt_pre(optical)
        sar_feat = self.sar_pre(sar)
        
        # 动态路由
        routed_feat = self.router(opt_feat, sar_feat)
        
        # 跨模态重组
        fused = self.recombiner(routed_feat, sar_feat)
        
        # 残差连接
        return fused #+ self.skip(torch.cat([opt_feat, sar_feat], dim=1))

from torchvision.ops import deform_conv2d
class DeformConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x
    
# class DeformConv2d(nn.Module):
#     """
#     可变形卷积V2版本 (Deformable Convolution v2)
#     输入参数:
#         in_channels (int): 输入通道数
#         out_channels (int): 输出通道数
#         kernel_size (int): 卷积核大小（假设为方形核）
#         stride (int): 步长，默认为1
#         padding (int): 填充，默认为0
#         dilation (int): 空洞率，默认为1
#         groups (int): 分组卷积数，默认为1
#         bias (bool): 是否使用偏置，默认为False
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, 
#                  stride=1, padding=0, dilation=1, groups=1, bias=False):
#         super().__init__()
        
#         # 基础参数
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
        
#         # 生成offset的卷积层
#         self.offset_conv = nn.Conv2d(
#             in_channels, 
#             2 * kernel_size * kernel_size,  # 每个位置(x,y)偏移量
#             kernel_size=3, 
#             padding=1,
#             stride=stride,
#             bias=True
#         )
        
#         # 调制因子卷积层 (Deformable Conv v2新增)
#         self.modulator_conv = nn.Conv2d(
#             in_channels,
#             kernel_size * kernel_size,  # 每个位置的调制因子
#             kernel_size=3,
#             padding=1,
#             stride=stride,
#             bias=True
#         )
        
#         # 常规权重参数
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels//groups, 
#                         kernel_size, kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
            
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
#         # Offset和Modulator卷积初始化
#         nn.init.constant_(self.offset_conv.weight, 0)
#         nn.init.constant_(self.offset_conv.bias, 0)
#         nn.init.constant_(self.modulator_conv.weight, 0)
#         nn.init.constant_(self.modulator_conv.bias, 0.5)  # 初始调制因子为0.5

#     def forward(self, x, offset=None):
#         # 如果未提供offset，则自动生成
#         if offset is None:
#             # 生成offset和modulator
#             offset = self.offset_conv(x)
#             modulator = 2 * torch.sigmoid(self.modulator_conv(x))  # [0,2]范围
            
#         # 获取输入特征图尺寸
#         B, C, H, W = x.size()
#         kh, kw = self.kernel_size, self.kernel_size
        
#         # 生成采样网格
#         y_range = torch.arange(0, H*self.stride, self.stride, 
#                               device=x.device, dtype=x.dtype)
#         x_range = torch.arange(0, W*self.stride, self.stride, 
#                               device=x.device, dtype=x.dtype)
#         y, x = torch.meshgrid(y_range, x_range)
#         y = y.contiguous().view(-1)
#         x = x.contiguous().view(-1)
#         xy = torch.stack([x, y], dim=1)  # 基础坐标 (中心点)
        
#         # 生成kernel的偏移网格
#         yy, xx = torch.meshgrid(
#             torch.arange(-(kh//2), kh//2+1, dtype=x.dtype, device=x.device),
#             torch.arange(-(kw//2), kw//2+1, dtype=x.dtype, device=x.device)
#         )
#         kernel_offset = torch.cat([xx.reshape(-1), yy.reshape(-1)], 0)  # (2*kh*kw,)
#         # kernel_offset = kernel_offset[None, None, :, None]  # (1,1,2*kh*kw,1)
        
#         # 计算最终采样位置
#         offset = offset.view(B, 2*kh*kw, -1)  # (B, 2*Kh*Kw, H*W)
#         offset = offset.permute(0, 2, 1).contiguous().view(B*H*W, 2*kh*kw)
#         sampling_loc = xy[:, None, :] + offset + kernel_offset  # (B*H*W, Kh*Kw, 2)
        
#         # 调整采样位置到[-1,1]范围
#         sampling_loc[:, :, 0] = (sampling_loc[:, :, 0] - (W-1)*0.5) / (W-1)*2
#         sampling_loc[:, :, 1] = (sampling_loc[:, :, 1] - (H-1)*0.5) / (H-1)*2
        
#         # 双线性插值采样
#         x_sampled = F.grid_sample(
#             x.view(1, B*C, H, W),  # 输入需要reshape为(1, B*C, H, W)
#             sampling_loc.view(B, H*W*kh*kw, 2), 
#             align_corners=True
#         ).view(B, C, H*W, kh*kw)
        
#         # 应用调制因子 (Deformable v2)
#         modulator = modulator.view(B, kh*kw, H*W).permute(0,2,1)  # (B, H*W, Kh*Kw)
#         x_sampled = x_sampled * modulator.unsqueeze(1)  # (B, C, H*W, Kh*Kw)
        
#         # 执行卷积操作
#         x_unfold = x_sampled.permute(0,1,3,2).reshape(B, C*kh*kw, H*W)
#         weight = self.weight.view(self.groups, -1, C//self.groups, kh*kw)
#         weight = weight.permute(0,2,3,1).reshape(-1, kh*kw, C//self.groups)
        
#         output = torch.matmul(x_unfold, weight).view(B, -1, H, W)
        
#         if self.bias is not None:
#             output += self.bias.view(1, -1, 1, 1)
            
#         return output

class Grad_edge(nn.Module):
    def __init__(self, kensize=5,):
        super(Grad_edge, self).__init__()
        self.k = kensize
        def creat_gauss_kernel(r=1):
            M_13 = np.concatenate([np.ones([r+1, 2*r+1]), np.zeros([r, 2*r+1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2 * r + 1]), np.ones([r+1, 2 * r + 1])], axis=0)

            M_11 = np.concatenate([np.ones([2*r+1, r+1]), np.zeros([2*r+1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2 * r + 1, r]), np.ones([2 * r + 1, r+1])], axis=1)
            return torch.from_numpy((M_13)).float(), torch.from_numpy((M_23)).float(), torch.from_numpy((M_11)).float(), torch.from_numpy((M_21)).float()
        M13, M23, M11, M21 = creat_gauss_kernel(self.k)
        weight_x1 = M11.view(1, 1, self.k*2+1, self.k*2+1)
        weight_x2 = M21.view(1, 1, self.k*2+1, self.k*2+1)
        weight_y1 = M13.view(1, 1, self.k*2+1, self.k*2+1)
        weight_y2 = M23.view(1, 1, self.k*2+1, self.k*2+1)
        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect") + 1e-2
        gx_1 = F.conv2d(
            x, self.weight_x1, bias=None, stride=1, padding=0, groups=1
        )
        gx_2 = F.conv2d(
            x, self.weight_x2, bias=None, stride=1, padding=0, groups=1
        )
        gy_1 = F.conv2d(
            x, self.weight_y1, bias=None, stride=1, padding=0, groups=1
        )
        gy_2 = F.conv2d(
            x, self.weight_y2, bias=None, stride=1, padding=0, groups=1
        )
        gx_rgb = torch.log((gx_1) / (gx_2))
        gy_rgb = torch.log((gy_1) / (gy_2))
        # norm_rgb = torch.cat([gx_rgb,gy_rgb],dim=1)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        # return torch.tensor(norm_rgb.nan_to_num().cpu().numpy().astype(np.uint8), dtype=torch.float, device=gy_rgb.device)
        return norm_rgb.nan_to_num().type(torch.uint8).type(x.dtype)

class HOGLayerC(nn.Module):
    """Generate hog feature for each batch images. This module is used in
    Maskfeat to generate hog feature. This code is borrowed from.
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>
    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16,
                 norm_out: bool = False,
                 in_channels: int = 1) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.in_channels = in_channels
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)
        self.norm_out = norm_out

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()


    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.
        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        hw = x.shape[-2], x.shape[-1]  
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=self.in_channels)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=self.in_channels)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=x.dtype,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        # out = out.sum(dim=2, keepdim=True)  # Sum across bins to get a single channel

        if self.norm_out:
            out = F.normalize(out, p=2, dim=2)
        if out.shape[1] == 1: # single channel for SAR images
            out = out.squeeze(1)
        
        out = nn.functional.interpolate(out, hw, mode='bilinear')

        return out


# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            if u.device == torch.device('cpu'):
                dtype = 'cpu'
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                u = u.to(device)
                delta = delta.to(device)
                A = A.to(device)
                B = B.to(device)
                C = C.to(device)
                D = D.to(device)
                delta_bias = delta_bias.to(device)
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
                out = out.to(dtype)
                x = x.to(dtype)
            else:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None
        
    class CrossScan_multimodal(torch.autograd.Function):  # wchao
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, HW = x_rgb.shape
            H = int(math.sqrt(HW))
            ctx.shape = (B, C, HW)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * HW))
            x_rgb = x_rgb.view(B, C, H, H)
            x_e = x_e.view(B, C, H, H)
            xs_fuse[:, 0] = torch.concat([x_rgb, x_e], dim=2).view(B, C, 2 * HW)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, HW = ctx.shape
            # L = 2 * HW
            H = int(math.sqrt(HW))
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
            # get B, d, H*W
            ys = ys.view(B, C, 2*H, H)
            y0, y1 = torch.split(ys, H, dim=2)
            return y0.view(B, -1, HW), y1.view(B, -1, HW)
            # return ys[:, :, 0:HW].view(B, -1, HW), ys[:, :, HW:2*HW].view(B, -1, HW) 
         
    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            H = int(math.sqrt(L//2))
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
            ys = ys.view(B, -1, 2*H, H)
            y0, y1 = torch.split(ys, H, dim=2)
            return y0.view(B, -1, L//2), y1.view(B, -1, L//2)
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            # return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape

            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2*L))
            H = int(math.sqrt(L))
            x1 = x1.view(B, C, H, H)
            x2 = x2.view(B, C, H, H)

            xs[:, 0] = torch.concat([x1, x2], dim=2).view(B, C, 2 * L)

    
            # xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2*L)
            return xs, None, None

    # class CrossScan_multimodal(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
    #         # B, C, H, W -> B, 2, C, 2 * H * W
    #         B, C, HW = x_rgb.shape
    #         ctx.shape = (B, C, HW)
    #         xs_fuse = x_rgb.new_empty((B, 2, C, 2 * HW))
    #         xs_fuse[:, 0] = torch.concat([x_rgb, x_e], dim=2)
    #         xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
    #         return xs_fuse

    #     @staticmethod
    #     def backward(ctx, ys: torch.Tensor):
    #         # out: (b, 2, d, l)
    #         B, C, HW = ctx.shape
    #         L = 2 * HW
    #         ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
    #         # get B, d, H*W
    #         return ys[:, :, 0:HW].view(B, -1, HW), ys[:, :, HW:2*HW].view(B, -1, HW) 
         
    # class CrossMerge_multimodal(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, ys: torch.Tensor):
    #         B, K, D, L = ys.shape
    #         # ctx.shape = (H, W)
    #         # ys = ys.view(B, K, D, -1)
    #         ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
    #         # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
    #         return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
    #     @staticmethod
    #     def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
    #         # B, D, L = x.shape
    #         # out: (b, k, d, l)
    #         # H, W = ctx.shape
    #         B, C, L = x1.shape
    #         xs = x1.new_empty((B, 2, C, 2*L))
    #         xs[:, 0] = torch.cat([x1, x2], dim=2)
    #         xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
    #         xs = xs.view(B, 2, C, 2*L)
    #         return xs, None, None

    def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def selective_scan_1d(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)
        
        # y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def cross_selective_scan_multimodal_k1(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
        nrows = 1
        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L//2] + ys[:, 0, :, L//2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)
        
        return y

    def cross_selective_scan_multimodal_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, HW = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * HW

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
        nrows = 1
        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, HW -> B, 2, C, 2 * HW
       
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*HW)
        
        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, HW, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, HW, -1)

        y_rgb = out_norm1(y_rgb.type(x_rgb.dtype)).to(x_rgb.dtype)
        y_e = out_norm2(y_e.type(x_rgb.dtype)).to(x_e.dtype)
        
        return y_rgb, y_e

DEV = False
class ConMB_SS2D(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_act = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx_act = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

            self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs,
            )

            self.conv1d_modalx = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
     
    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        y_rgb_act = self.act(self.in_proj_act(x_rgb))
        y_e_act = self.act(self.in_proj_modalx_act(x_e))
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)

        if self.d_conv > 1:
            # x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            # x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_trans = x_rgb.permute(0, 2, 1).contiguous()
            x_e_trans = x_e.permute(0, 2, 1).contiguous()
            x_rgb_conv = self.act(self.conv1d(x_rgb_trans)) # (b, d, hw)
            x_e_conv = self.act(self.conv1d_modalx(x_e_trans)) # (b, d, hw)

            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv) # b, d, hw -> b, hw, d
            # SE to get attention, scale
            b, d, hw = x_rgb_trans.shape
            # x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            # x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            # x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1).permute(0, 2, 1).contiguous() # b, 1, 1, d
            # x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1).permute(0, 2, 3, 1).contiguous() 
            # y_rgb = y_rgb * x_e_exitation
            # y_e = y_e * x_rgb_exitation
            y_rgb = y_rgb*y_rgb_act
            y_e = y_e*y_e_act
        #     y = torch.concat([y_rgb, y_e], dim=-1)
        # out = self.dropout(self.out_proj(y))
        y_rgb = self.dropout(self.out_proj(y_rgb))
        y_e = self.dropout(self.out_proj1(y_e))
        return y_rgb, y_e

class TSGBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        
        # self.mlp_branch = mlp_ratio > 0
        # if self.mlp_branch:
        #     self.norm2 = norm_layer(hidden_dim)
        #     mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        #     self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs,
        )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.CMA_ssm = Cross_Mamba_Attention_SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, HW, D = x_rgb.shape
        if self.d_conv > 1:
            # x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            # x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            # x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            # x_e_conv = self.act(self.conv2d(x_e_trans)) # (b, d, h, w)
            # x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            # x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
            # y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv) 
            # # to b, d, h, w
            # y_rgb = y_rgb.view(B, H, W, -1)
            # y_e = y_e.view(B, H, W, -1)
            x_rgb_trans = x_rgb.permute(0, 2, 1).contiguous()
            x_e_trans = x_e.permute(0, 2, 1).contiguous()
            x_rgb_conv = self.act(self.conv1d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv1d(x_e_trans)) # (b, d, h, w)
            x_rgb_conv = rearrange(x_rgb_conv, "b d hw -> b hw d")
            x_e_conv = rearrange(x_e_conv, "b d hw -> b hw d")
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv) 
            # to b, d, h, w
            y_rgb = y_rgb.view(B, HW, -1)
            y_e = y_e.view(B, HW, -1)

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e

class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x  


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, num_heads=None, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    

class MambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim,
        d_state,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ConMB_SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def forward(self, x):
        x_rgb = x[0] 
        x_e = x[1]
        B, C, H, W = x_rgb.shape
        x_rgb = x_rgb.flatten(2).transpose(1, 2)
        x_e = x_e.flatten(2).transpose(1, 2)
        x0, x1 = self.op(x_rgb, x_e)
        x0 = self.drop_path(x0)
        x1 = self.drop_path(x1)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        x0 = x0.transpose(1, 2).view(B, C, H, W)
        x1 = x1.transpose(1, 2).view(B, C, H, W)
        return x0, x1

        # x_e = x_e.flatten(2).transpose(1, 2)
        # x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        # if self.mlp_branch:
        #     x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        # x = x.transpose(1, 2).view(B, C, H, W)
        # return x

    # def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
    #     '''
    #     B C H W, B C H W -> B C H W
    #     '''
    #     if self.use_checkpoint:
    #         return checkpoint.checkpoint(self._forward, x_rgb, x_e)
    #     else:
    #         return self._forward(x_rgb, x_e)

    
class CMIM(nn.Module): # Cross-modal Mamba Interaction Module (CMIM)
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim,
        d_state,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ConMB_SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def forward(self, x):
        x_rgb = x[0] 
        x_e = x[1]
        B, C, H, W = x_rgb.shape
        x_rgb = x_rgb.flatten(2).transpose(1, 2)
        x_e = x_e.flatten(2).transpose(1, 2)
        x0, x1 = self.op(x_rgb, x_e)
        x0 = self.drop_path(x0)
        x1 = self.drop_path(x1)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        x0 = x0.transpose(1, 2).view(B, C, H, W)
        x1 = x1.transpose(1, 2).view(B, C, H, W)
        return x0, x1
    

