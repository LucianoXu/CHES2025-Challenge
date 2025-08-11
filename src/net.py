from typing import Callable, Literal, Iterable, Optional, Tuple, Union
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle




# denoising function
def notch_many_fft(
    x: Union[torch.Tensor, "np.ndarray"],
    fs: float,
    freqs: Iterable[float],
    Q: float = 40.0,                 # Q ≈ f0 / (-3 dB bandwidth)
    taper_ratio: float = 0.5,        # 0..1, fraction of half-band used for cosine skirts
    atten_db: float = 60.0,          # depth of the notch (in dB)
    mix: float = 1.0,                # 1.0 = full notch, <1.0 = wet/dry mix
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Narrow-band noise removal using smooth frequency-domain notches.

    Args:
        x: 1D (T,) or 2D (N, T) real signal. torch tensor or numpy array.
        fs: Sample rate in Hz.
        freqs: Iterable of center frequencies to notch (Hz).
        Q: Quality factor; -3 dB bandwidth BW ≈ f0 / Q. Larger Q => narrower notch.
        taper_ratio: Cosine taper fraction of the half-band. 0 = brick-wall, 0.5 = pure cosine.
        atten_db: Notch depth in dB (amplitude domain). 60 dB means ~1e-3 gain at center.
        mix: Wet/dry mix. y = mix * filtered + (1-mix) * original.
        device: Torch device. If None, inferred from x or defaults to CPU.

    Returns:
        y : filtered signal with same shape/type as input.
    """
    # --- Normalize input to torch tensor (float32) and batch shape (B, T) ---
    is_numpy = False
    try:
        import numpy as _np
        if not torch.is_tensor(x):
            is_numpy = True
            x = torch.from_numpy(_np.asarray(x))
    except Exception:
        pass

    if device is None:
        device = x.device if torch.is_tensor(x) else torch.device("cpu")

    assert isinstance(x, torch.Tensor)
    
    x = x.to(device).float()
    if x.ndim == 1:
        x = x[None, :]  # (1, T)

    B, T = x.shape
    n_r = T // 2 + 1

    # --- rFFT and frequency bins ---
    X = torch.fft.rfft(x, dim=-1)  # (B, n_r)
    # rFFT bin frequencies: f_k = k * fs / T
    f_bins = torch.arange(n_r, device=device, dtype=x.dtype) * (fs / T)

    # --- Build smooth multiplicative mask: start from ones (pass-through) ---
    mask = torch.ones(n_r, device=device, dtype=x.dtype)

    # Precompute center attenuation (amplitude gain)
    att = 10.0 ** (-atten_db / 20.0)

    for f0 in freqs:
        # Skip invalid or out-of-band centers
        if not (0.0 < f0 < fs * 0.5):
            continue

        # Half-bandwidth by Q: H = BW/2 ≈ f0 / (2Q)
        H = float(f0) / (2.0 * float(Q))
        if H <= 0:
            continue

        # Split half-band into a flat "core" and cosine "skirts"
        taper = max(taper_ratio * H, 0.0)
        core  = max(H - taper, 0.0)

        # Distance from the center frequency
        delta = (f_bins - f0).abs()

        if taper > 0:
            # Smooth gain profile:
            #   delta <= core: gain = att
            #   core < delta < core+taper: cosine ramp from att -> 1
            #   delta >= core+taper: gain = 1
            t = ((delta - core) / taper).clamp(0.0, 1.0)          # 0..1
            r = 0.5 - 0.5 * torch.cos(torch.pi * t)               # 0..1 (cosine ease)
            gain = att + (1.0 - att) * r                          # att..1
        else:
            # Brick-wall notch within |delta| <= H
            gain = torch.where(delta <= H, torch.tensor(att, device=device, dtype=x.dtype),
                                           torch.tensor(1.0, device=device, dtype=x.dtype))

        mask = mask * gain  # combine multiple notches multiplicatively

    # --- Apply mask and inverse rFFT ---
    Xf = X * mask[None, :]
    y = torch.fft.irfft(Xf, n=T, dim=-1)

    # Wet/dry mix
    if mix != 1.0:
        y = mix * y + (1.0 - mix) * x

    # Restore original shape/type
    y = y.squeeze(0) if B == 1 else y
    if is_numpy:
        y = y.cpu().numpy()

    return y



# -------- Weights --------
def fracdiff_weights(d: float, lags: int, *, device=None, dtype=None) -> torch.Tensor:
    """
    Vectorized weights for (1 - L)^d up to given lags.
    w_k = (-1)^k * C(d, k),  C(d,k) = Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1))

    Args:
        d: fractional order (real, typical 0<d<1)
        lags: include lags 0..lags  (kernel length = lags+1)

    Returns:
        1D tensor of shape (lags+1,)
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")

    # compute in float64 for stability, then cast back
    work_dtype = torch.float64
    k = torch.arange(lags + 1, device=device, dtype=work_dtype)
    d_t = torch.as_tensor(d, device=device, dtype=work_dtype)

    # log-binomial via log-gamma for stability
    lg = torch.lgamma
    log_w_abs = lg(d_t + 1) - lg(k + 1) - lg(d_t - k + 1)
    sign = torch.where((k % 2) == 0, 1.0, -1.0)  # (-1)^k
    w = sign * torch.exp(log_w_abs)

    # cast to requested dtype (default float32)
    return w.to(dtype if dtype is not None else torch.float32)


# -------- Fractional differencing --------
def fractional_diff(x: torch.Tensor,
                    d: float,
                    *,
                    lags: int = 128,
                    center: bool = False) -> torch.Tensor:
    """
    Apply fractional differencing along the last dimension using causal convolution.

    Args:
        x: input (..., T)
        d: fractional order
        lags: number of lags to include (kernel length = lags+1)
        center: remove mean along time after differencing

    Returns:
        y with the same shape as x
    """
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    *batch, T = x.shape
    device, dtype = x.device, x.dtype

    # build kernel (causal: past-to-present); conv1d is xcorr so we flip
    w = fracdiff_weights(d, lags, device=device, dtype=dtype)       # (L,)
    kernel = w.flip(0).view(1, 1, -1)                               # (1,1,L)

    # explicit causal padding on the left so output length is T
    x_flat = x.reshape(-1, 1, T)                                    # (N,1,T)
    x_pad  = F.pad(x_flat, (kernel.size(-1) - 1, 0))                # left pad
    y = F.conv1d(x_pad, kernel)                                     # (N,1,T)
    y = y.reshape(*batch, T)

    if center:
        y = y - y.mean(dim=-1, keepdim=True)
    return y


# -------- Optional: pick lags automatically by tail cutoff --------
def suggest_lags(d: float, tol: float = 1e-4, max_lags: int = 10000) -> int:
    """
    Choose smallest lags so that |w_k| < tol for all k > lags (crude but practical).
    Uses stable recurrence; stops early.

    Note: For small tol and d~0.5, this can still be large. Prefer fixed lags (e.g., 256).
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    w_prev = 1.0
    lags = 0
    k = 1
    while k <= max_lags:
        w_k = -w_prev * (d - k + 1.0) / k
        if abs(w_k) < tol:
            break
        w_prev = w_k
        lags = k
        k += 1
    return lags

class TimestepWindow(nn.Module):
    '''
    The module that cuts the input tensor along the first dimension, according to the start indices and the window size.
    '''
    def __init__(self, model_args: dict):
        super(TimestepWindow, self).__init__()
        self.start_indices = model_args["start_indices"]
        self.window_size = model_args["window_size"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: input tensor, shape (N, T, C) or (N, T)
        Returns:
            cut tensor, shape (N, window_size, C) or (N, window_size)
        '''
        if len(x.shape) == 3:
            return x[:, self.start_indices:self.start_indices + self.window_size, :]
        else:
            return x[:, self.start_indices:self.start_indices + self.window_size]

class MLP(nn.Module):
    def __init__(self, model_args: dict):
        super(MLP, self).__init__()
        self.model_args = model_args
        self.input_dim = model_args["input_dim"]
        self.hidden_dims = model_args["hidden_dims"]
        self.output_dim = model_args["output_dim"]
        self.activation = model_args["activation"]

        self.std_m : dict[str, torch.Tensor]

        if self.model_args["fractional_diff"] is not None:
            self.input_dim += model_args["input_dim"]

        # the input window
        if self.model_args["input_window"]:
            self.window = TimestepWindow(self.model_args['input_window_args'])

        if self.model_args["gated"]:
            self.gate = nn.Parameter(torch.ones(self.model_args["input_dim"]))


        self.layers = nn.ModuleList()

        for layer_index in range(0, len(self.hidden_dims)):
            if layer_index == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer_index]))
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer_index - 1], self.hidden_dims[layer_index]))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())

        if len(self.hidden_dims) == 0:
            self.last_layer = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.last_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, x):
        '''
        Input: (N, T)
        '''

        # denoising
        x = notch_many_fft(
            x, 
            **{
                "fs": 1000,
                "freqs": [20, 25, 30, 40, 60, 80],
                "Q": 100,
                "atten_db": 20
            }, 
            device=x.device,
        )

        # standardization
        x = (x - self.std_m['mean_attack']) + self.std_m['mean_profiling_raw_key_127']
        x = (x - self.std_m['mean_profiling']) / self.std_m['std_profiling']

        x = x.unsqueeze(-1) # (N, T) -> (N, T, 1)

        if self.model_args["input_window"]:
            x = self.window(x)

        if self.model_args["gated"]:
            x = x * self.gate.unsqueeze(-1)

        x = x.flatten(start_dim=1)  # (N, T, 1) -> (N, T)

        # calculate the fractional difference
        if self.model_args["fractional_diff"] is not None:
            diff = fractional_diff(x, d=self.model_args["fractional_diff"], lags=16)
            x = torch.cat((x, diff), dim=-1)  # (N, T) -> (N, 2*T)

        for layer in self.layers:
            x = layer(x)
        x = self.last_layer(x)
        return x


class CNN(nn.Module):

    @staticmethod
    def complete_cnn_hp(model_args: dict):
        '''
        Automatically decide the pooling size and kernel number according to the specification, and calculate the corresponding coefficients.
        The results are stored in the `model_args` dictionary.
        '''

        pooling_type = model_args["pooling_type"]

        if pooling_type == "max_pool":
            cal_fun_pool = cal_size_after_maxpool1d
        elif pooling_type == "average_pool":
            cal_fun_pool = cal_size_after_avgpool1d
        else:
            raise ValueError("Invalid pooling type: {}".format(pooling_type))

        # the size for kernels in each layer
        kernel_sizes: list[int] = model_args["kernel_sizes"]
        layer_strides: list[int] = model_args["layer_strides"]
        input_size = model_args["input_size"]
        input_dim = model_args["input_dim"]
        output_size = model_args["output_size"]
        output_dim = model_args["output_dim"]
        padding = model_args["padding"]

        num_kernels : list[int] = []
        pooling_sizes: list[int] = []  # the strides in each layer, (n) strides in total
        layer_sizes: list[int] = []    # the size of data in each layer, (n+1) sizes in total, the first is the input size
        layer_dims: list[int] = []  # the dimension of data in each layer, (n+1) dims in total, the first is the input dim

        # actually, the number of kernels is the sublist of the layer_dims

        layer_sizes.append(input_size)
        layer_dims.append(input_dim)
        for i in range(1, len(kernel_sizes)+1):
            # calculate the size after convolution
            size_after_conv = cal_size_after_conv1d(layer_sizes[i-1], kernel_sizes[i-1], layer_strides[i-1], padding)
            # calculate the ideal output size after pooling
            ideal_output_size = round(input_size * (output_size / input_size) ** (i/len(kernel_sizes)))
            # calculate the real pooling size
            real_pooling_size = round(size_after_conv / ideal_output_size)
            if real_pooling_size < 1:
                real_pooling_size = 1

            real_output_size = cal_fun_pool(size_after_conv, real_pooling_size, real_pooling_size)

            pooling_sizes.append(real_pooling_size)
            layer_sizes.append(real_output_size)

            # calculate the dimension of the layer
            dim = round(input_dim * (output_dim / input_dim) ** (i/len(kernel_sizes)))
            layer_dims.append(dim)
            num_kernels.append(dim)

        model_args["num_kernels"] = num_kernels
        model_args["pooling_sizes"] = pooling_sizes
        model_args["layer_sizes"] = layer_sizes
        model_args["layer_dims"] = layer_dims




    def __init__(self, model_args: dict):
        super(CNN, self).__init__()

        self.free_cache = False

        self.model_args = model_args.copy()

        # complete the CNN parameters
        CNN.complete_cnn_hp(self.model_args)

        self.num_layers = len(self.model_args["kernel_sizes"])
        self.activation = self.model_args["activation"]
        self.pooling_type = self.model_args["pooling_type"]

        # the input window
        if self.model_args["input_window"]:
            self.window = TimestepWindow(self.model_args['input_window_args'])

        if self.model_args["gated"]:
            self.gate = nn.Parameter(torch.ones(self.model_args["input_size"]))

        self.layers = nn.ModuleList()
        
        for layer_index in range(0, self.num_layers):
            #Convolution layer
        
            self.layers.append(nn.Conv1d(
                in_channels=self.model_args["layer_dims"][layer_index], 
                out_channels=self.model_args["layer_dims"][layer_index + 1], 
                kernel_size=self.model_args["kernel_sizes"][layer_index],
                stride=self.model_args["layer_strides"][layer_index], 
                padding=self.model_args["padding"])
            )

            #Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())

            #Pooling Layer
            if self.model_args["pooling_sizes"][layer_index] != 1:
                if self.pooling_type == "max_pool":
                    self.layers.append(nn.MaxPool1d(
                        kernel_size=self.model_args['pooling_sizes'][layer_index], 
                        stride=self.model_args['pooling_sizes'][layer_index])
                    )
                elif self.pooling_type == "average_pool":
                    self.layers.append(nn.AvgPool1d(
                        kernel_size=self.model_args['pooling_sizes'][layer_index], 
                        stride=self.model_args['pooling_sizes'][layer_index])
                    )

            #BatchNorm
            # self.layers.append(nn.BatchNorm1d(self.model_args["layer_dims"][layer_index + 1]))

        # global average pooling
        if self.model_args["global_pooling_type"] == "average_pool":
            self.layers.append(nn.AdaptiveAvgPool1d(1))  # (N, C, T) -> (N, C, 1)
        elif self.model_args["global_pooling_type"] == "max_pool":
            self.layers.append(nn.AdaptiveMaxPool1d(1))
        elif self.model_args["global_pooling_type"] == "none":
            pass
        else:
            raise ValueError("Invalid global pooling type: {}".format(self.model_args["global_pooling_type"]))
            
        #MLP
        if self.model_args["mlp_head"]:
            self.mlp = MLP(self.model_args["mlp_head_args"])

    def forward(self, x):

        if self.model_args["input_window"]:
            x = self.window(x)

        if self.model_args["gated"]:
            x = x * self.gate.unsqueeze(-1)

        x = x.transpose(1, 2)  # (N, T, C) -> (N, C, T)

        for layer in self.layers:
            x = layer(x)
            if self.free_cache:
                torch.cuda.empty_cache()

        # x : (N, C, T)
        x = x.flatten(start_dim=1).unsqueeze(-1)  # (N, C, T) -> (N, D, 1)

        if self.model_args["mlp_head"]:
            x = self.mlp(x)

        return x


# Helper functions for CNN hyperparameter creation
def cal_size_after_conv1d(n_sample_points, kernel_size, stride, padding = 0, dilation = 1):
        L_in = n_sample_points
        L_out = math.floor(((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)+1)
        return L_out

def cal_size_after_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_size_after_avgpool1d(n_sample_points, kernel_size, stride, padding=0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size ) / stride) + 1)
    return L_out
