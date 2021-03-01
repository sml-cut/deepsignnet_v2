"""
Implementation of Dense and Convolutional Bayesian layers employing LWTA activations and IBP in pyTorch, as described in
Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.
ReLU and Linear activations are also implemented.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

import torch, math
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F
from signjoey.utils import concrete_sample, kumaraswamy_sample, bin_concrete_sample
import weakref
import numpy as np
import time
class DenseBayesian(Module):
    """
    Class for a Bayesian Dense Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """
    instances = []
    def __init__(self, input_features, output_features, competitors,
                 activation, prior_mean, prior_scale, temperature = 0.67, ibp = False, bias=True,name='NoName'):
        """

        :param input_features: int: the number of input_features
        :param output_features: int: the number of output features
        :param competitors: int: the number of competitors in case of LWTA activations, else 1
        :param activation: str: the activation to use. 'relu', 'linear' or 'lwta'
        :param prior_mean: float: the prior mean for the gaussian distribution
        :param prior_scale: float: the prior scale for the gaussian distribution
        :param temperature: float: the temperature of the relaxations.
        :param ibp: boolean: flag to use the IBP prior.
        :param bias: boolean: flag to use bias.
        """

        super(DenseBayesian, self).__init__()
        self.name=name
        self.__class__.instances.append(self)
        self.input_features = input_features
        self.output_features = output_features
        self.blocks = output_features // competitors
        self.competitors = competitors
        self.activation = activation
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.prior_conc1 = torch.tensor(2.)
        self.prior_conc0 = torch.tensor(2.)
        self.temperature = temperature
        self.ibp = ibp
        self.bias  = bias
        self.tau = 1e-2
        self.training = True
        
        self.outscale=1.0#/torch.sqrt(torch.tensor(self.input_features*1.0))
        
        self.posterior_mean = Parameter(torch.Tensor(output_features, input_features))
        # posterior unnormalized scale. Needs to be passed by softplus
        self.posterior_un_scale = Parameter(torch.Tensor(output_features, input_features))
        self.register_buffer('weight_eps', None)

        if activation == 'lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2')
                self.competitors = 2
                self.blocks = output_features // 2
            if output_features % competitors != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} units in groups of {}..'.format(output_features, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')



        if bias:
            self.bias_mean = Parameter(torch.Tensor(output_features))
            self.bias_un_scale = Parameter(torch.Tensor(output_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_un_scale', None)
            self.register_buffer('bias_eps', None)

        if ibp:
            self.conc1 = Parameter(torch.Tensor(output_features))
            self.conc0 = Parameter(torch.Tensor(output_features))
            self.t_pi = Parameter(torch.Tensor(input_features, output_features))
            print(self.t_pi.shape)
        else:
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)


        self.reset_parameters()
        print('\n ###############################\n')
        print('Bayesian layers -> IBP =',ibp,' activation=',activation)
        
        print('\n ###############################\n')
        time.sleep(2)
        
    def reset_parameters(self):
        """
        Initialization function for all the parameters of the Dense Bayesian Layer.

        :return: null
        """

        # can change this to uniform with std or something else
        #stdv = 1. / math.sqrt(self.posterior_mean.size(1))
        #self.posterior_mean.data.uniform_(-stdv, stdv)

        # original init
        init.kaiming_uniform_(self.posterior_mean, a = math.sqrt(5))
        self.posterior_un_scale.data.fill_(-0.2*self.prior_scale)
        print('first:','std:',torch.mean(F.softplus(self.posterior_un_scale,beta=10)),' - ', torch.std(F.softplus(self.posterior_un_scale,beta=10)) )
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.posterior_mean)
            bound = 1. / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound, bound)

            self.bias_un_scale.data.fill_(-5.*self.prior_scale)

        if self.ibp:
            self.conc1.data.fill_(self.blocks)
            self.conc0.data.fill_(1.)

            init.uniform_(self.t_pi, -5, 1.)
            #init.uniform_(self.t_pi, 0., 1.)


    def forward(self, input):
        """
        Override the default forward function to implement the Bayesian layer.

        :param input: torch tensor: the input data

        :return: torch tensor: the output of the current layer
        """

        if self.training:
            # use the reparameterization trick
            weights_sample = self.posterior_mean +F.softplus(self.posterior_un_scale,beta=10) * torch.randn_like(self.posterior_un_scale)

            if self.bias:
                bias_sample = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
            else:
                bias_sample = None

            if self.ibp:
                conc1_soft = F.softplus(self.conc1,beta=10)
                conc0_soft = F.softplus(self.conc0,beta=10)


                q_u = kumaraswamy_sample(conc1_soft, conc0_soft, input.size(1))
                pi = torch.cumprod(q_u, -1)

                self.pi = pi
                # posterior probabilities

                t_pi_sigmoid = torch.sigmoid(self.t_pi)

                z = bin_concrete_sample(t_pi_sigmoid, self.temperature)


                weights = z.T*weights_sample

            else:
                weights = weights_sample

            bias = bias_sample

        else:

            weights = self.posterior_mean
            bias = self.bias_mean

            if self.ibp:
                z = bin_concrete_sample(torch.sigmoid(self.t_pi), 0.01)#, self.temperature, hard = True)
                z = (torch.sigmoid(self.t_pi) > self.tau) * z
                weights = z.T*weights

        weights = self.posterior_mean
        bias = self.bias_mean        
        if np.random.uniform()<0.001:
            
         print('mean:',torch.mean(torch.abs(self.posterior_mean)),' - ','std:',torch.mean(F.softplus(self.posterior_un_scale,beta=10)))
            
        if self.activation == 'linear':
            return F.linear(input, weights, bias)*self.outscale
        elif self.activation == 'relu':
            return F.relu(F.linear(input, weights, bias)*self.outscale)
        elif self.activation == 'softsign':
            return F.softsign(F.linear(input, weights, bias)*self.outscale)
        elif self.activation == 'lwta':
            return self.lwta_activation(F.linear(input, weights, bias)*self.outscale, not self.training)
        else:
            raise ValueError(self.activation + " is not implemented..")


    def lwta_activation(self, input, hard = False, eps = 1e-5):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        logits = torch.reshape(input, [-1, self.blocks, self.competitors])

        probs = F.softmax(logits, -1) + eps
        probs /= torch.sum(probs, -1, keepdims = True)
        xi = concrete_sample(probs, temperature = self.temperature, hard = hard)
        self.probs_xi = probs
        out = (logits * xi).reshape(input.shape)
        if np.random.uniform()<0.01:
            print('probs',probs)

        return out


    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return "prior_mean = {}, prior_scale = {}, input_features = {}, output_features = {}, bias = {}".format(
            self.prior_mean, self.prior_scale, self.input_features, self.output_features, self.bias
        )



###########################################################################
##################### BAYESIAN CONV IMPLEMENTATION ########################
###########################################################################
class ConvBayesian(Module):
    """
    Class for a Bayesian Conv Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, competitors,
                 activation, prior_mean, prior_scale, temperature = 0.01, ibp = False, bias=True):
        """

        :param in_channels: int: the number of input channels
        :param out_channels: int: the number of the output channels
        :param kernel_size: int: the size of the kernel
        :param stride: int: the stride of the kernel
        :param padding: int: padding for the convolution operation
        :param competitors: int: the number of competitors in case of LWTA activations, else 1
        :param activation: str: the activation to use. 'relu', 'linear' or 'lwta'
        :param prior_mean: float: the prior mean for the gaussian distribution
        :param prior_scale: float: the prior scale for the gaussian distribution
        :param temperature: float: the temperature for the employed relaxations
        :param ibp: boolean: flag to use the IBP prior
        :param bias: boolean; flag to use bias
        """

        super(ConvBayesian, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.competitors = competitors
        self.blocks = out_channels // competitors
        self.activation = activation
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.temperature = temperature
        self.ibp = ibp
        self.bias  = bias
        self.training = True

        if activation == 'lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2')
                self.competitors = 2
                self.blocks = out_channels // self.competitors
            if out_channels % competitors != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} feature maps in groups of {}..'.format(out_channels, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.competitors = 1
                self.blocks = out_channels // self.competitors

        self.posterior_mean = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                     self.kernel_size, self.kernel_size))
        # posterior unnormalized scale. Needs to be passed by softplus
        self.posterior_un_scale = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                     self.kernel_size, self.kernel_size))
        self.register_buffer('weight_eps', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_channels))
            self.bias_un_scale = Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_un_scale', None)
            self.register_buffer('bias_eps', None)

        if ibp:
            self.conc1 = Parameter(torch.Tensor(self.blocks))
            self.conc0 = Parameter(torch.Tensor(self.blocks))
            self.t_pi = Parameter(torch.Tensor(self.blocks))

        else:
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)



        self.reset_parameters()


    def reset_parameters(self):
        """
        Initialization function for all the parameters of the Dense Bayesian Layer.

        :return: null
        """

        # can change this to uniform with std or something else
        #stdv = 1. / math.sqrt(self.posterior_mean.size(1))
        #self.posterior_mean.data.uniform_(-stdv, stdv)

        # original init
        init.kaiming_uniform_(self.posterior_mean, a = math.sqrt(5))
        self.posterior_un_scale.data.fill_(-5.*self.prior_scale)

        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.posterior_mean)
            bound = 1. / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound, bound)

            self.bias_un_scale.data.fill_(-5.*self.prior_scale)

        if self.ibp:
            self.conc1.data.fill_(3.)
            self.conc0.data.fill_(1.)
            init.uniform_(self.t_pi, -5, 1.)


    def forward(self, input):
        """
        Overrride the forward pass to match the necessary computations.

        :param input: torch tensor: the input to the layer.

        :return: torch tensor: the output of the layer after activation
        """

        if self.training:
            # use the reparameterization trick
            weights_sample = self.posterior_mean + F.softplus(self.posterior_un_scale) * torch.randn_like(self.posterior_un_scale)

            if self.bias:
                bias_sample = self.bias_mean + F.softplus(self.bias_un_scale) * torch.randn_like(self.bias_un_scale)
            else:
                bias_sample = None

            if self.ibp:
                conc1_soft = F.softplus(self.conc1)
                conc0_soft = F.softplus(self.conc0)

                q_u = kumaraswamy_sample(conc1_soft, conc0_soft, conc1_soft.size(0))
                pi = torch.cumprod(q_u, -1)

                self.pi = pi

                # posterior probabilities
                t_pi_sigmoid = torch.sigmoid(self.t_pi)

                z = concrete_sample(t_pi_sigmoid, self.temperature)
                z = z.repeat(self.competitors).view(-1, 1, 1,1)

                weights = z*weights_sample
            else:
                weights = weights_sample

            bias = bias_sample

        else:
            weights = self.posterior_mean
            bias = self.bias_mean

            if self.ibp:
                z = concrete_sample(torch.sigmoid(self.t_pi), self.temperature, hard = True)
                weights *= z.repeat(self.competitors)
                #print(weights.shape)
                #print(z.shape)

        if self.activation == 'linear':
            return F.conv2d(input, weights, bias, stride = self.stride, padding = self.padding)
        elif self.activation == 'relu':
            return F.relu(F.conv2d(input, weights, bias, stride = self.stride, padding = self.padding))
        elif self.activation == 'lwta':
            return self.lwta_activation(F.conv2d(input, weights, bias, stride = self.stride, padding = self.padding))
        
        else:
            raise ValueError(self.activation + " is not implemented..")

    def lwta_activation(self, input, hard = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        logits = torch.reshape(input, [-1, self.blocks, self.competitors, input.size(-1), input.size(-1)])

        probs = F.softmax(logits, 2) + 1e-10
        probs /= torch.sum(probs, 2, keepdims = True)
        xi = concrete_sample(probs, temperature = self.temperature, hard = hard, axis = 2)
        self.probs_xi = probs
        out = (logits * xi).reshape(input.shape)
        #print(logits*xi)

        return out


    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return "prior_mean = {}, prior_scale = {}, input_features = {}, output_features = {}, bias = {}".format(
            self.prior_mean, self.prior_scale, self.in_channels, self.out_channels, self.bias
        )