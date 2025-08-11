import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F

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

        if self.model_args["fractional_diff"] is not None:
            self.input_dim += model_args["input_dim"]

        # the input window
        if self.model_args["input_window"]:
            self.window = TimestepWindow(self.model_args['input_window_args'])

        if self.model_args["gated"]:
            self.gate = nn.Parameter(torch.ones(self.model_args["input_dim"]))


        self.layers = nn.ModuleList()

        for layer_index in range(0, len(self.hidden_dims)):
            layer = nn.Sequential()

            if layer_index == 0:
                layer.append(nn.Linear(self.input_dim, self.hidden_dims[layer_index]))
            else:
                layer.append(nn.Linear(self.hidden_dims[layer_index - 1], self.hidden_dims[layer_index]))

            if self.activation == 'relu':
                layer.append(nn.ReLU())
            elif self.activation == 'selu':
                layer.append(nn.SELU())
            elif self.activation == 'tanh':
                layer.append(nn.Tanh())
            elif self.activation == 'elu':
                layer.append(nn.ELU())

            # add the dropout layer
            if self.model_args["dropout_rate"] is not None and layer_index >= len(self.hidden_dims) - 2:
                layer.append(nn.Dropout(self.model_args["dropout_rate"]))

            self.layers.append(layer)

        if len(self.hidden_dims) == 0:
            self.last_layer = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.last_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, x):
        '''
        Input: (N, T, 1)
        '''
        if self.model_args["input_window"]:
            x = self.window(x)

        if self.model_args["gated"]:
            x = x * self.gate.unsqueeze(-1)

        x = x.flatten(start_dim=1)  # (N, T, 1) -> (N, T)

        # calculate the fractional difference
        if self.model_args["fractional_diff"] is not None:
            diff = fractional_diff(x, d=self.model_args["fractional_diff"], lags=16)
            x = torch.cat((x, diff), dim=-1)  # (N, T) -> (N, 2*T)

        x_prev = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.model_args["residual_connection"] is not None and i % self.model_args["residual_connection"] == 0:
                if x_prev is None:
                    x_prev = x
                else:
                    x = x + x_prev  # residual connection
                    x_prev = x

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
            if self.free_cache and torch.cuda.is_available():
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
