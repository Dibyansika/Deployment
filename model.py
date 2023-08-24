import torch
from torch import nn
from torch import Tensor, nn
from math import ceil
from qiskit_ibm_runtime import IBMBackend, Options
from typing import Any, List, Optional, Tuple
from zipfile import ZipFile
import numpy as np
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.circuit import Parameter
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from efficientnet_pytorch import EfficientNet
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import torch
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import (Estimator, IBMBackend, Options,
                                QiskitRuntimeService, RuntimeJob)

model_path = '/tmp/.cache/torch/checkpoints/efficientNet.pth'
# # Convolution, Batch Normalization, and Activation Class

# class ConvBnAct(nn.Module):
    
#     def __init__(self, n_in, n_out, kernel_size = 3, stride = 1, 
#                  padding = 0, groups = 1, bn = True, act = True,
#                  bias = False
#                 ):
        
#         super(ConvBnAct, self).__init__()
        
#         self.conv = nn.Conv2d(n_in, n_out, kernel_size = kernel_size,
#                               stride = stride, padding = padding,
#                               groups = groups, bias = bias
#                              )
#         self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
#         self.activation = nn.SiLU() if act else nn.Identity()
        
#     def forward(self, x):
        
#         x = self.conv(x)
#         x = self.batch_norm(x)
#         x = self.activation(x)
        
#         return x
    
# ''' Squeeze and Excitation Block '''

# class SqueezeExcitation(nn.Module):
    
#     def __init__(self, n_in, reduced_dim):
#         super(SqueezeExcitation, self).__init__()
        
        
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(n_in, reduced_dim, kernel_size=1),
#             nn.SiLU(),
#             nn.Conv2d(reduced_dim, n_in, kernel_size=1),
#             nn.Sigmoid()
#         )
       
#     def forward(self, x):
        
#         y = self.se(x)
        
#         return x * y

# # Stochastic Depth Module

# class StochasticDepth(nn.Module):
    
#     def __init__(self, survival_prob = 0.8):
#         super(StochasticDepth, self).__init__()
        
#         self.p =  survival_prob
        
#     def forward(self, x):
        
#         if not self.training:
#             return x
        
#         binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        
#         return torch.div(x, self.p) * binary_tensor
    
# # Residual Bottleneck Block with Expansion Factor = N as defined in 
# # Mobilenet-V2 paper with Squeeze and Excitation Block and Stochastic Depth.
# class MBConvN(nn.Module):
    
#     def __init__(self, n_in, n_out, kernel_size = 3, 
#                  stride = 1, expansion_factor = 6,
#                  reduction = 4, # Squeeze and Excitation Block
#                  survival_prob = 0.8 # Stochastic Depth
#                 ):
        
#         super(MBConvN, self).__init__()
        
#         self.skip_connection = (stride == 1 and n_in == n_out) 
#         intermediate_channels = int(n_in * expansion_factor)
#         padding = (kernel_size - 1)//2
#         reduced_dim = int(n_in//reduction)
        
#         self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, intermediate_channels, kernel_size = 1)
#         self.depthwise_conv = ConvBnAct(intermediate_channels, intermediate_channels,
#                                         kernel_size = kernel_size, stride = stride, 
#                                         padding = padding, groups = intermediate_channels
#                                        )
#         self.se = SqueezeExcitation(intermediate_channels, reduced_dim = reduced_dim)
#         self.pointwise_conv = ConvBnAct(intermediate_channels, n_out, 
#                                         kernel_size = 1, act = False
#                                        )
#         self.drop_layers = StochasticDepth(survival_prob = survival_prob)
        
#     def forward(self, x):
        
#         residual = x
        
#         x = self.expand(x)
#         x = self.depthwise_conv(x)
#         x = self.se(x)
#         x = self.pointwise_conv(x)
        
#         if self.skip_connection:
#             x = self.drop_layers(x)
#             x += residual
        
#         return x


# import numpy as np
# from qiskit import QuantumCircuit, Aer, assemble
# from qiskit.circuit import Parameter

# class MyQuantumCircuit:
#     def __init__(self, n_qubits, shots):
#         self._circuit = QuantumCircuit(n_qubits)
#         all_qubits = [i for i in range(n_qubits)]

#         self.theta_params = [Parameter(f'theta{i}') for i in range(7)]

#         self._circuit.h(all_qubits)
#         self._circuit.barrier()
#         self._circuit.ry(self.theta_params[0], all_qubits)
#         self._circuit.cz(0,1)
#         self._circuit.cz(1,2)
#         self._circuit.ry(self.theta_params[1], 0)
#         self._circuit.ry(self.theta_params[2], 1)
#         self._circuit.ry(self.theta_params[3], 2)
#         self._circuit.cz(0,1)
#         self._circuit.cz(1,2)
#         self._circuit.ry(self.theta_params[4], 0)
#         self._circuit.ry(self.theta_params[5], 1)
#         self._circuit.ry(self.theta_params[6], 2)
#         self._circuit.measure_all()

#         self.backend = Aer.get_backend("qasm_simulator")
#         self.shots = shots

#     def run(self, thetas):
#         parameter_binds = {param: value for param, value in zip(self.theta_params, thetas.tolist())}
# #         parameter_binds=thetas.tolist()
#         bound_circuit = self._circuit.bind_parameters(parameter_binds)

#         qobj = assemble(bound_circuit,
#                         shots=(self.shots))

#         job = self.backend.run(qobj)
#         result = job.result()

#         counts = result.get_counts(bound_circuit)

#         expects = np.zeros(8)
#         for k, key in enumerate(['000', '001', '010', '011', '100', '101', '110', '111']):
#             perc = counts.get(key, 0) / self.shots
#             expects[k] = perc
#         return expects
    
# class QuantumFunction(torch.autograd.Function):
#     """custom autograd function that uses a quantum circuit"""

#     @staticmethod
#     def forward(
#         ctx,
#         batch_inputs: Tensor,
#         qc: MyQuantumCircuit,
#     ) -> Tensor:
#         """forward pass computation"""
# #         print(batch_inputs.tolist())
#         ctx.save_for_backward(batch_inputs)
#         ctx.qc = qc
#         ctx.shift = torch.pi / 2,
# #         result=[]
# #         for parameter_values in batch_inputs.tolist():
# #             result.append(qc.run(torch.tensor(parameter_values)).tolist())
# #         return torch.tensor(result)#qc.run(batch_inputs)
#         flattened_tensor = batch_inputs.flatten()
#         qc.run(flattened_tensor)
#         return batch_inputs

#     @staticmethod
#     def backward(
#         ctx,
#         grad_output: Tensor
#     ):
#         """backward pass computation using parameter shift rule"""
#         batch_inputs = ctx.saved_tensors[0]
#         qc = ctx.qc

#         shifted_inputs_r = torch.empty(batch_inputs.shape)
#         shifted_inputs_l = torch.empty(batch_inputs.shape)

#         # loop over each input in the batch
#         for i, _input in enumerate(batch_inputs):

#             # loop entries in each input
#             for j in range(len(_input)):

#                 # compute parameters for parameter shift rule
#                 d = torch.zeros(_input.shape)
#                 d[j] = ctx.shift
#                 shifted_inputs_r[i, j] = _input + d
#                 shifted_inputs_l[i, j] = _input - d

#         # run gradients in batches
#         exps_r=[]
#         exps_l=[]
#         print(shifted_inputs_r.tolist())
#         for parameter_values in shifted_inputs_r.tolist():
#             exps_r.append(qc.run(parameter_values).tolist())
        
#         for parameter_values in shifted_inputs_l.tolist():
#             exps_l.append(qc.run(torch.tensor(parameter_values)).tolist())

#         return (torch.tensor(exps_r) - torch.tensor(exps_l)).float() * grad_output.float(), None, None

# class QuantumLayer(torch.nn.Module):
#     """a neural network layer containing a quantum function"""

#     def __init__(
#         self,
#         n_qubits: int,
# #         estimator: Estimator,
#         shots:int
#     ):
#         super().__init__()
#         self.qc = MyQuantumCircuit(
#             n_qubits=n_qubits,
# #             estimator=estimator,
#             shots=shots
#         )

#     def forward(self, xs: Tensor) -> Tensor:
#         """forward pass computation"""

#         result = QuantumFunction.apply(xs, self.qc)

#         # if xs.shape[0] == 1:
#         # return result.view((1, 1))
#         return result

# '''Efficient-net Class'''
# from qiskit.primitives import Estimator
# estimator = Estimator(options=Options().__dict__)
# class EfficientNet(nn.Module):
    
#     '''Generic Efficient net class which takes width multiplier, Depth multiplier, and Survival Prob.'''
    
#     def __init__(self, width_mult = 1, depth_mult = 1, 
#                  dropout_rate = 0.2, num_classes = 5):
#         super(EfficientNet, self).__init__()
        
#         last_channel = ceil(1280 * width_mult)
#         self.quantum = QuantumLayer(3,10000)
#         self.features = self._feature_extractor(width_mult, depth_mult, last_channel)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(last_channel, num_classes)
#         )
        
#     def forward(self, x):
#         x = self.quantum(x)
#         x = self.features(x)
#         x = self.avgpool(x)
        
#         x = self.classifier(x.view(x.shape[0], -1))
        
#         return x
    
        
#     def _feature_extractor(self, width_mult, depth_mult, last_channel):
        
#         channels = 4*ceil(int(32*width_mult) / 4)
#         layers = [ConvBnAct(3, channels, kernel_size = 3, stride = 2, padding = 1)]
#         in_channels = channels
        
#         kernels = [3, 3, 5, 3, 5, 5, 3]
#         expansions = [1, 6, 6, 6, 6, 6, 6]
#         num_channels = [16, 24, 40, 80, 112, 192, 320]
#         num_layers = [1, 2, 2, 3, 3, 4, 1]
#         strides =[1, 2, 2, 2, 1, 2, 1]
        
#         # Scale channels and num_layers according to width and depth multipliers.
#         scaled_num_channels = [4*ceil(int(c*width_mult) / 4) for c in num_channels]
#         scaled_num_layers = [int(d * depth_mult) for d in num_layers]

        
#         for i in range(len(scaled_num_channels)):
#             layers += [MBConvN(in_channels if repeat==0 else scaled_num_channels[i], 
#                                scaled_num_channels[i],
#                                kernel_size = kernels[i],
#                                stride = strides[i] if repeat==0 else 1, 
#                                expansion_factor = expansions[i]
#                               )
#                        for repeat in range(scaled_num_layers[i])
#                       ]
#             in_channels = scaled_num_channels[i]
        
#         layers.append(ConvBnAct(in_channels, last_channel, kernel_size = 1, stride = 1, padding = 0))
    
#         return nn.Sequential(*layers)
def EfficientNetB4(pretrained=False):
    """Constructs a EfficientNetB0 model for FastAI.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 1 }) ## Regressor
    if pretrained:
        model_state = torch.load(model_path)
        # load original weights apart from its head
        if '_fc.weight' in model_state.keys():
            model_state.pop('_fc.weight')
            model_state.pop('_fc.bias')
            res = model.load_state_dict(model_state, strict=False)
            assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        else:
            # A basic remapping is required
            from collections import OrderedDict
            mapping = { i:o for i,o in zip(model_state.keys(), model.state_dict().keys()) }
            mapped_model_state = OrderedDict([
                (mapping[k], v) for k,v in model_state.items() if not mapping[k].startswith('_fc')
            ])
            res = model.load_state_dict(mapped_model_state, strict=False)
            print(res)
    return model

class KappaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5]
        # define score function:
        self.func = self.quad_kappa
    
    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == 'Tensor':
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.FloatTensor(preds).view(-1)

        for i,pred in enumerate(y_hat):
            if   pred < coef[0]: y_hat[i] = 0
            elif pred < coef[1]: y_hat[i] = 1
            elif pred < coef[2]: y_hat[i] = 2
            elif pred < coef[3]: y_hat[i] = 3
            else:                y_hat[i] = 4
        return y_hat.int()
    
    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds)
        return cohen_kappa_score(y, y_hat, weights='quadratic')

    def fit(self, preds, y):
        ''' maximize quad_kappa '''
        print('Early score:', self.quad_kappa(preds, y))
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter':1000, 'fatol':1e-20, 'xatol':1e-20})
        print(opt_res)
        self.coef = opt_res.x
        print('New score:', self.quad_kappa(preds, y))

    def forward(self, preds, y):
        ''' the pytorch loss function '''
        return torch.tensor(self.quad_kappa(preds, y))