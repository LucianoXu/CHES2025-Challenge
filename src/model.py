import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.num_layers = model_args["layers"]
        self.input_dim = model_args["input_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.output_dim = model_args["output_dim"]
        self.activation = model_args["activation"]

        # the input window
        if self.model_args["input_window"]:
            self.window = TimestepWindow(self.model_args['input_window_args'])

        if self.model_args["gated"]:
            self.gate = nn.Parameter(torch.ones(self.model_args["input_dim"]))


        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())

        self.last_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        '''
        Input: (N, T, 1)
        '''
        if self.model_args["input_window"]:
            x = self.window(x)

        if self.model_args["gated"]:
            x = x * self.gate.unsqueeze(-1)

        x = x.transpose(1, 2)  # (N, T, 1) -> (N, 1, T)
        for layer in self.layers:
            x = layer(x)
        x = self.last_layer(x) #F.softmax()
        x = x.squeeze(1)
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
            self.layers.append(nn.BatchNorm1d(self.model_args["layer_dims"][layer_index + 1]))

        # global average pooling
        self.layers.append(nn.AdaptiveAvgPool1d(1))  # (N, C, T) -> (N, C, 1)
            
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

        # x : (N, C, 1)

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



#############################

def create_hyperparameter_space(model_type):
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                                   "lr": random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                                    "optimizer": random.choice( ["RMSprop", "Adam"]),
                                                    "layers": random.randrange(1, 8, 1),
                                                    "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                                    "activation": random.choice(  ["relu", "selu", "elu", "tanh"]),
                                                    "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                                                }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(1000, 1001, 100),
                                              "lr":random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                              "optimizer":random.choice(["RMSprop", "Adam"]),
                                              "layers": random.randrange(1, 8, 1),
                                              "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                              "activation": random.choice( ["relu", "selu", "elu", "tanh"]),
                                              "kernel_initializer": random.choice( ["random_uniform", "glorot_uniform", "he_uniform"]),
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes":random.choice(  [2,4,6,8,10]), #size == strides
                                              "conv_layers": random.choice( [1,2,3,4]),
                                              "filters": random.choice( [4,8,12,16]),
                                              "kernels": random.choice( [i for i in range(26,53,2)]), #strides = kernel/2
                                              "padding": random.choice(  [0,4,8,12,16]),
                                        }

        return search_space