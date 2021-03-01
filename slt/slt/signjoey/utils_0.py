"""
Implementation of utility functions for the Dense and Convolutional Bayesian layers employing LWTA activations and IBP in pyTorch, as described in
Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

###################################################
################## DIVERGENCES ####################
###################################################

def kl_divergence_concrete(prior_probs, posterior_probs):
    """
    Compute the kl divergence for two concrete distributions.

    :param prior_probs: torch tensor: The probabilities of the prior distribution.
    :param posterior_probs: torch tensor: the probabilities of the posterior distribution.

    :return: scalar: the kl divergence
    """

    kl_loss = - (torch.log(prior_probs) - torch.log(posterior_probs))

    return torch.sum(torch.mean(kl_loss, 0))


def kl_divergence_bin_concrete(prior_probs, posterior_probs, temp, eps = 1e-8):
    """
    Compute the kl divergence for two binary concrete distributions.

    :param prior_probs: torch tensor: The probabilities of the prior distribution.
    :param posterior_probs: torch tensor: the probabilities of the posterior distribution.

    :return: scalar: the kl divergence
    """

    device = torch.device("cuda" if posterior_probs.is_cuda else "cpu")

    U = torch.rand(posterior_probs.shape).to(device)
    L = torch.log(U + eps) - torch.log(1. - U + eps)
    X = torch.sigmoid(( L + torch.log(posterior_probs))/ temp)

    return X.sum()


def kl_divergence_kumaraswamy_(prior_a, prior_b, a, b, sample):
    """
    KL divergence for the Kumaraswamy distribution.

    :param prior_a: torch tensor: the prior a concentration
    :param prior_b: torch tensor: the prior b concentration
    :param a: torch tensor: the posterior a concentration
    :param b: torch tensor: the posterior b concentration
    :param sample: a sample from the Kumaraswamy distribution

    :return: scalar: the kumaraswamy kl divergence
    """
   
    q_log_prob = kumaraswamy_log_pdf(a, b, sample)
    p_log_prob = beta_log_pdf(prior_a, prior_b, sample)
   
    return - (p_log_prob - q_log_prob).sum()

def kl_divergence_kumaraswamy(prior_a, prior_b, a, b, sample):
    """
    KL divergence for the Kumaraswamy distribution.

    :param prior_a: torch tensor: the prior a concentration
    :param prior_b: torch tensor: the prior b concentration
    :param a: torch tensor: the posterior a concentration
    :param b: torch tensor: the posterior b concentration
    :param sample: a sample from the Kumaraswamy distribution

    :return: scalar: the kumaraswamy kl divergence
    """
    x=torch.linspace(0.001, 0.999, 2048).to('cuda')
    x=torch.rand(sample.shape).to('cuda')
    x=x*0.999+0.001
  #  print(a)
  #  print('\n\n ***** \n')
   # print(sample.shape)
    q_log_prob = kumaraswamy_log_pdf(a, b, x)
    p_log_prob = beta_log_pdf(prior_a, prior_b, x)
   
    y= - ( torch.exp(q_log_prob)*(p_log_prob - q_log_prob))
    out=torch.trapz(y,x)
    
  #  print(out)
    return  out.sum()

def kl_divergence_normal(prior_mean, prior_scale, posterior_mean, posterior_scale):
    """
     Compute the KL divergence between two Gaussian distributions.

    :param prior_mean: torch tensor: the mean of the prior Gaussian distribution
    :param prior_scale: torch tensor: the scale of the prior Gaussian distribution
    :param posterior_mean: torch tensor: the mean of the posterior Gaussian distribution
    :param posterior_scale: torch tensor: the scale of the posterior Gaussian distribution

    :return: scalar: the kl divergence between the prior and posterior distributions
    """

    device = torch.device("cuda" if posterior_mean.is_cuda else "cpu")


    prior_scale_normalized = F.softplus(torch.Tensor([prior_scale]).to(device),beta=10)
    posterior_scale_normalized = F.softplus(posterior_scale,beta=10)

    kl_loss = -0.5 + torch.log(prior_scale_normalized) - torch.log(posterior_scale_normalized) \
                + (posterior_scale_normalized ** 2 + (posterior_mean - prior_mean)**2) / (2 * prior_scale_normalized**2)


    return kl_loss.sum()
    

# scan for deeper children  
def get_children(model):
    model_children = list(model.children())
    for child in model_children:
        model_children+=get_children(child)
    return model_children


def model_kl_divergence_loss(model, kl_weight = 1.,layers_list=[]):
    """
    Compute the KL losses for all the layers of the considered model.

    :param model: nn.Module extension implementing the model with our custom layers
    :param kl_weight: scalar: the weight for the KL divergences

    :return: scalar: the KL divergence for all the layers of the model.
    """

    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    # get the layers as a list
    model_children = get_children(model)

            
 
    if len(layers_list)==0:
        layers_list=model_children
   
    for layer in layers_list:
       
        if hasattr(layer, 'prior_mean'):

            kl = kl_divergence_normal(layer.prior_mean, layer.prior_scale, layer.posterior_mean, layer.posterior_un_scale)

            kl_sum += kl
            n += len(layer.posterior_mean.view(-1))

            if layer.bias:
                kl = kl_divergence_normal(layer.prior_mean, layer.prior_scale, layer.bias_mean, layer.bias_un_scale)
                kl_sum += kl
                n += len(layer.bias_mean.view(-1))

            if layer.ibp:

                # kl for the binary indicators
                kl = kl_divergence_bin_concrete(torch.ones_like(layer.t_pi)/layer.blocks, torch.sigmoid(layer.t_pi), layer.temperature)
               
                kl_sum += kl
                n += len(layer.t_pi.view(-1))

                # kl for the sticks
                kl = kl_divergence_kumaraswamy(layer.prior_conc1, layer.prior_conc0, layer.conc1, layer.conc0, layer.pi)
                kl_sum += kl
                
                n += len(layer.prior_conc1.view(-1))


            # if we want to use samples for the full concrete kl divergence, save a sample in the layer
            if layer.activation == 'lwta' and hasattr(layer, 'probs_xi'):
           
                kl = kl_divergence_concrete(torch.ones_like(layer.probs_xi)*0.5, layer.probs_xi)
                kl_sum += 100*kl
                print(kl)
                n += layer.probs_xi.size(1) * layer.probs_xi.size(2)

    epsilon=0.00000001
    return kl_weight * kl_sum[0] / (n[0]+epsilon)



###########################################
########## DISTRIBUTIONS ##################
###########################################

def kumaraswamy_log_pdf(a,b, x):
    """
    The kumaraswamy log pdf.

    :param a: torch tensor: the a concentration of the distribution
    :param b: torch tensor: the b concentration of the distribution
    :param x: torch tensor: a sample from the kumaraswamy distribution

    :return: torch tensor: the log pdf of the kumaraswamy distribution
    """

    a = F.softplus(a)
    b = F.softplus(b)

    return torch.log(a) + torch.log(b) + (a - 1.)*torch.log(x) + (b - 1.)*torch.log(1. - x**a)


def beta_log_pdf(a, b, x):
    """

    The log pdf of the beta distribution.

    :param a: torch tensor: the a concentration of the distribution
    :param b: torch tensor: the b concentration of the distribution
    :param x: torch tensor: a sample from the Beta/Kumaraswamy distribution

    :return: torch tensor: the log pdf of the beta distribution
    """

    device = torch.device("cuda" if x.is_cuda else "cpu")
    log_pdf = torch.distributions.beta.Beta(a.to(device), b.to(device)).log_prob(x)

    return log_pdf


def kumaraswamy_sample(conc1, conc0, batch_shape):
    """
    Sample from the Kumaraswamy distribution given the concentrations

    :param conc1: torch tensor: the a concentration of the distribution
    :param conc0: torch tensor: the b concentration of the distribution
    :param batch_shape: scalar: the batch shape for the samples

    :return: torch tensor: a sample from the Kumaraswamy distribution
    """

    device = torch.device("cuda" if conc1.is_cuda else "cpu")

    x = (0.01 - 0.99) * torch.rand(batch_shape, conc1.size(0)).to(device) + 0.99
    q_u = (1-(1-x)**(1./conc0))**(1./conc1)

    return q_u

def bin_concrete_sample(probs, temperature, hard = False, eps = 1e-8):
    """"
    Sample from the binary concrete distribution
    """

    device = torch.device("cuda" if probs.is_cuda else "cpu")
    U = torch.rand(probs.shape)
    L = Variable(torch.log(U + eps) - torch.log(1. - U + eps)).to(device)
    X = torch.sigmoid((L + torch.log(probs)) / temperature)

    return X

def concrete_sample(probs, temperature, hard = False, eps = 1e-7, axis = -1):
    """
    Sample from the concrete relaxation.

    :param probs: torch tensor: probabilities of the concrete relaxation
    :param temperature: float: the temperature of the relaxation
    :param hard: boolean: flag to draw hard samples from the concrete distribution
    :param eps: float: eps to stabilize the computations
    :param axis: int: axis to perform the softmax of the gumbel-softmax trick

    :return: a sample from the concrete relaxation with given parameters
    """

    device = torch.device("cuda" if probs.is_cuda else "cpu")

    U = torch.rand(probs.shape)
    G = - torch.log(- torch.log(U + eps) + eps)
    t = (torch.log(probs) + Variable(G).to(device)) / temperature

    y_soft = F.softmax(t, axis) + eps
    y_soft /= torch.sum(y_soft, axis, keepdims = True)

    if hard:
        #_, k = y_soft.data.max(axis)
        _, k = probs.data.max(axis)
        shape = probs.size()

        if len(probs.shape) == 2:
            y_hard = torch.log(probs).new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        else:
            y_hard = torch.log(probs).new(*shape).zero_().scatter_(-1, k.view(-1, probs.size(1), 1), 1.0)


        y = Variable(y_hard - y_soft.data) + y_soft

    else:
        y = y_soft

    return y


###############################################
################ CONSTRAINTS ##################
###############################################
class parameterConstraints(object):
    """
    A class implementing the constraints for the parameters of the layers.
    """

    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'posterior_un_scale'):
            scale = module.posterior_un_scale
            scale = scale.clamp(-7., 1000.)
            module.posterior_un_scale.data = scale

        if hasattr(module, 'bias_un_scale'):
            scale = module.bias_un_scale
            scale = scale.clamp(-7., 1000.)
            module.bias_un_scale.data = scale

        if hasattr(module, 'conc1') and module.conc1 is not None:
            conc1 = module.conc1
            conc1 = conc1.clamp(-6., 1000.)
            module.conc1.data = conc1

        if hasattr(module, 'conc0') and module.conc0 is not None:
            conc0 = module.conc0
            conc0 = conc0.clamp(-6., 1000.)
            module.conc0.data = conc0

        if hasattr(module, 't_pi') and module.t_pi is not None:
            t_pi = module.t_pi
            t_pi = t_pi.clamp(-7, 600.)
            module.t_pi.data = t_pi

