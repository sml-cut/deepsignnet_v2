"""
Implementation of Dense and Convolutional Bayesian layers employing LWTA activations and IBP in pyTorch, as described in
Panousis  et al., Nonparametric Bayesian Deep Networks with Local Competition.
ReLU and Linear activations are also implemented.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

import torch, math
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from signjoey.utils import concrete_sample, kumaraswamy_sample, bin_concrete_sample, kl_divergence_kumaraswamy
import numpy as np
import weakref
import pandas as pd





class LWTAMASK2(Module):
    def __init__(self,size,kl_w=1000):
        super(LWTAMASK2, self).__init__()
        competitors=2
        self.U=competitors
        self.kl_w=kl_w
        self.ln= DenseBayesian(size, 2, competitors = 1,
                                activation = 'linear', prior_mean=0, prior_scale=1. )
                
        if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 2
                
        self.posterior_mean = Parameter(torch.Tensor(1, 1))
        self.mask= Parameter(torch.Tensor(1,1,size,2))
       
                
    def forward(self, input, temp = 1.67, hard = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """
        if not self.training:
            self.temperature= 0.01
        else:
            self.temperature= 2.01
        
        self.n=0
        self.loss=0
        kl=0
        
      
        
        logits=torch.unsqueeze(input,-1)
        ran = torch.normal(logits*0,0.01)  
        logits= torch.cat((logits,ran),-1)
        
        if not self.training:
            xi = concrete_sample(self.mask*10, temperature = self.temperature, hard = hard)
        else:
            xi = concrete_sample(self.mask, temperature = self.temperature, hard = hard)
        out = logits*xi
        if self.training:
            out=out[:,:,:,0]+out[:,:,:,1]
        else :
            out=out[:,:,:,0]
        
    
        if self.training:
            q = F.softmax(self.mask, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))
            log_p=log_q*0
            log_p[:,:,:,0]=-9
            log_p[:,:,:,1]=0
            
            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
            self.loss+=kl
            self.loss*=self.kl_w
        if not self.training and np.random.uniform()<0.05:
            M=xi.to('cpu').detach().numpy()
            M=np.round(M)
            M=np.sum(np.round(M[:,:,:,0]))/np.sum(np.round(M))
           
            print('Mask 2:',np.round(M*100,2),'%')
        return out




class LWTAMASK(Module):
    def __init__(self,size,competitors,hidden=False):
        super(LWTAMASK, self).__init__()
        self.U=competitors
        self.size=size
        self.hidden=hidden
       # self.ln= DenseBayesian(512, 1, competitors = 1,
          #                      activation = 'linear', prior_mean=0, prior_scale=1.,bias=False )
        self.ln=torch.nn.Linear(512,1,bias=False)
        if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 2
                
        self.posterior_mean = Parameter(torch.Tensor(1, 1))
       
        
    def forward(self, input,m, temp = 1.67, hard = False,shift=0):
        
        if not self.training:
            self.temperature= 0.01
        else:
            self.temperature= 0.67
       
        
        hidden_dim=4*(m.shape[1]//self.U)+4
        
        self.n=0
        self.loss=0
        kl=0
        logits=input
        mask = self.ln(m)
  

        K=hidden_dim//self.U    
            
        
        if shift>0:
            mask = torch.roll(mask,shift,1)
        
        
        mask2 = torch.zeros(mask.shape[0], hidden_dim, 1).cuda()
        
        mask2[:,:mask.shape[1],:] = mask
       
        if shift<0:
            
            mask2  = torch.cat([ mask2[:,i::self.U,:] for i in range(self.U)],axis=-2)
            mask2 = torch.reshape(mask2, [-1,K, self.U])
        mask2 = torch.reshape(mask2, [-1,hidden_dim//self.U, self.U])
        
        if not self.training:
            xi = concrete_sample(mask2, temperature = self.temperature, hard = hard)
        else:
            xi = concrete_sample(mask2, temperature = self.temperature, hard = hard)
      
        if shift<0:
            xi=xi.reshape([-1,hidden_dim,1])
            xo=xi*0
            
            for i in range(self.U):
                xo[:,i::self.U,0]=xi[:,K*i:K*(i+1),0]
            xi=xo
        else:
            xi = torch.reshape(xi,[-1,hidden_dim,1])
        xi=xi[:,:mask.shape[1],:]
        
        
        if shift>0:
            xi = torch.roll(xi,-shift,-2)
        out = xi*input
       
        
       
        if self.training:
            q = F.softmax(mask2, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
            self.loss=kl*100
        else:
            if np.random.uniform()<0.001 and (not self.hidden):
                print('\n',xi,'\n')
        
        
        if self.hidden:
            return xi
        return out

class SuperMask(Module):
    
    def __init__(self,size,competitors):
        super(SuperMask, self).__init__()
        self.mask1=LWTAMASK(size,competitors,hidden=True)
        self.mask2=LWTAMASK(size,competitors,hidden=True)
    
    def forward(self,input,m):
        
        x1=self.mask1(m,m)
        x2=self.mask2(m,m,shift=-1)
         
        if self.training :
            p=2
         
        else : 
            p=16
     
        xi=torch.pow(torch.pow(x1+0.0001,p)+torch.pow(x2+0.0001,p),1/p)
        
       
        if not self.training and np.random.uniform()<0.001:
            print(( torch.round(100*xi[0]) ))

            
        out = input*xi
        return out
    
class RandomMask(Module):
    
    def __init__(self,size,a):
        super(RandomMask, self).__init__()
        self.a=a
    
    def forward(self,input,m):
        out=torch.zeros(m.shape).cuda()
        out[:,::self.a]=input[:,::self.a]+out[:,::self.a]
        
       
        return out
    
class LWTA(Module):
    def __init__(self,competitors):
        super(LWTA, self).__init__()
        self.U=competitors
        
        
                
        if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 2
                
        self.posterior_mean = Parameter(torch.Tensor(1, 1))
       
                
    def forward(self, input, temp = 1.67, hard = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """
        if not self.training:
            self.temperature= 0.01
        else:
            self.temperature= 2.01
        mod=input.shape[-1]%self.U
        if mod!=0:
            result = F.pad(input=input, pad=(0, self.U-mod, 0, 0), mode='constant', value=0)
            added=True
        else:
            result=input
            added=False
        self.K=result.shape[-1]//self.U
        self.n=0
        self.loss=0
        kl=0
        A=1
        if A==0:
            logits = torch.reshape(result, [-1,self.K, self.U])
            xi = concrete_sample(logits, temperature = self.temperature, hard = hard)
            out = logits*xi
            out = out.reshape(result.shape)
        elif A==1:
            logits=torch.unsqueeze(result,-1)
            ran = torch.normal(logits*0,1)
            
            lr= torch.cat((logits,ran),-1)
            logits=lr
            xi = concrete_sample(lr, temperature = temp, hard = hard)
            out = lr*xi
            return
        elif A==2:
            logits = torch.unsqueeze(result,-1)
            r=np.random.randint(-5,5)+np.random.randint(-5,5)+np.random.randint(-5,5)
            r1=torch.roll(logits,r,-2)
            
          
            logits=torch.cat((logits,r1),-1)
            xi = concrete_sample(logits, temperature = temp, hard = hard)
            out = logits*xi
            out = xi[:,:,0]
            logits = torch.squeeze(logits)
        
        
    
        if self.training:
            q = F.softmax(logits, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))
            log_p=log_q*0
            log_p[:,:,:,0]=0
            log_p[:,:,:,1]=1
            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
            self.loss+=kl
            self.loss*=10
        if added:
            out=out[:,:-(self.U-mod)]
        if np.random.uniform()<0.01:
            print(xi[0,:,1])
        if not self.training:
           pass #S print(torch.sum(torch.sign(abs(out)),axi=-2))
        return out


class DenseBayesian(Module):
    """
    Class for a Bayesian Dense Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """
    instances=weakref.WeakSet()
    ID=0
    def __init__(self, input_features, output_features, competitors,
                 activation, deterministic = False, temperature = 0.67, ibp = False, bias=True,prior_mean=1,prior_scale=1,kl_w=1.0,name=None,out_w=False,init_w=1.0):
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
        
        competitors=4
        self.ID=DenseBayesian.ID
        DenseBayesian.ID+=1
        self.name=name
        self.n=0.0001
        
        self.kl_w=kl_w
        self.init_w=init_w
        DenseBayesian.instances.add(self)
        self.input_features = input_features
        self.output_features = output_features
        self.K = output_features // competitors
        self.U = competitors
        self.activation = activation
        self.deterministic = deterministic

        self.temperature = 1.67#temperature
        self.ibp = ibp
        self.bias  = bias
        self.tau = 1e-2
        self.training = True
        self.out_wYN=out_w
        
        if out_w:
            self.out_w=Parameter(torch.Tensor(1))
      #  self.out_w.to('cuda')
        #################################
        #### DEFINE THE PARAMETERS ######
        #################################

        self.posterior_mean = Parameter(torch.Tensor(output_features, input_features))

        if not deterministic:
            # posterior unnormalized scale. Needs to be passed by softplus
            self.posterior_un_scale = Parameter(torch.Tensor(output_features, input_features))
            self.register_buffer('weight_eps', None)

        if activation == 'lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 4
                self.K = output_features // 4
            if output_features % self.U != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} units in groups of {}..'.format(output_features, competitors))
                
        elif activation == 'superlwta':
            
            if competitors  < 2:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 2
                self.K = output_features // 2
            if output_features % self.U != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} units in groups of {}..'.format(output_features, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.K = output_features
                self.U = 1

        #########################
        #### IBP PARAMETERS #####
        #########################
        if ibp:
            self.prior_conc1 = torch.tensor(1.)
            self.prior_conc0 = torch.tensor(1.)

            self.conc1 = Parameter(torch.Tensor(self.K))
            self.conc0 = Parameter(torch.Tensor(self.K))

            self.t_pi = Parameter(torch.Tensor(input_features, self.K))
        else:
            self.register_parameter('prior_conc1', None)
            self.register_parameter('prior_conc0', None)
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(output_features))

            if not deterministic:
                self.bias_un_scale = Parameter(torch.Tensor(output_features))
                self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            if not deterministic:
                self.register_parameter('bias_un_scale', None)
                self.register_buffer('bias_eps', None)


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
        #init.xavier_normal_(self.posterior_mean)
        init.kaiming_uniform_(self.posterior_mean, a = 0.01*math.sqrt(5))
        if not self.deterministic:
            self.posterior_un_scale.data.fill_(-0.1)

        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.posterior_mean)
            bound = 1. / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound*0.1*self.init_w, bound*0.1)
            #self.bias_mean.data.fill_(0.1)

            if not self.deterministic:
                self.bias_un_scale.data.fill_(-0.9)

        if self.ibp:
            self.conc1.data.fill_(2.)
            self.conc0.data.fill_(.5453)

            init.uniform_(self.t_pi, .1, 1.)
     
    def forward(self, input):
        """
        Override the default forward function to implement the Bayesian layer.

        :param input: torch tensor: the input data

        :return: torch tensor: the output of the current layer
        """
      
        layer_loss = 0.
        self.n=0
        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = F.softplus(self.posterior_un_scale,beta=10)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)

                #kl_weights = -0.5 * torch.mean(2*posterior_scale - torch.square(self.posterior_mean)
                #                               - posterior_scale ** 2 + 1)
                kl_weights = -0.5 * torch.sum(2*torch.log(posterior_scale) - torch.square(self.posterior_mean)
                                               - torch.square(posterior_scale) + 1)
                layer_loss += torch.sum(kl_weights)
                self.n += len(self.posterior_mean.view(-1))

            else:
                W = self.posterior_mean


            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z.T*W

                layer_loss += kl_sticks
                layer_loss += kl_z

            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
                    bias_kl = -0.5 * torch.sum(2*torch.log(F.softplus(self.bias_un_scale,beta=10)) - 
                                                   torch.square(self.bias_mean)
                                                   - torch.square(F.softplus(self.bias_un_scale,beta=10)) + 1)
                    self.n += len(self.bias_mean.view(-1))
                    layer_loss += torch.sum(bias_kl)
                else:
                    bias = self.bias_mean
            else:
                bias = None

        else:
            #posterior_scale = F.softplus(self.posterior_un_scale,beta=10)*0.01
           # W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)
            W = self.posterior_mean

            if self.bias:
                bias = self.bias_mean
            else:
                bias = None

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.01)
                W = z.T*W

        out = F.linear(input, W, bias)
        if self.out_wYN:
            out=out*torch.sigmoid(self.out_w).to('cuda')
            layer_loss=layer_loss+torch.sigmoid(self.out_w)
            if np.random.uniform()<0.001:
                print(torch.sigmoid(self.out_w))
        #if np.random.uniform()<0.001:
        #    print('\n\n\n',self.ID,' ',self.name)
        #    print(torch.min( F.softplus(self.posterior_un_scale,beta=10)), torch.max( torch.abs(self.posterior_mean) ))
        if self.activation == 'linear':
            self.loss = layer_loss
            self.loss*=self.kl_w
            
            return out

        elif self.activation == 'relu':
            self.loss = layer_loss
            self.loss*=self.kl_w
            return F.relu(out)

        elif self.activation == 'lwta':
            out, kl =  self.lwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            self.loss*=self.kl_w
            return out
        elif self.activation == 'new':
            out, kl =  self.new_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            self.loss*=self.kl_w
            return out
        elif self.activation == 'superlwta':
            out, kl =  self.superlwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            self.loss*=self.kl_w
            return out
        else:
            raise ValueError(self.activation + " is not implemented..")


    def indian_buffet_process(self, temp = 0.67):

        kl_sticks = kl_z = 0.
        z_sample = bin_concrete_sample(self.t_pi, temp)

        if not self.training:
            t_pi_sigmoid = torch.sigmoid(self.t_pi)
            mask = t_pi_sigmoid >self.tau
            z_sample = t_pi_sigmoid*mask

        z = z_sample.repeat(1, self.U)

        # compute the KL terms
        if self.training:

            a_soft = F.softplus(self.conc1)
            b_soft = F.softplus(self.conc0)

            q_u = kumaraswamy_sample(a_soft, b_soft, sample_shape = [self.t_pi.size(0), self.t_pi.size(1)])
            prior_pi = torch.cumprod(q_u, -1)

            q = torch.sigmoid(self.t_pi)
            log_q = torch.log(q + 1e-6)
            log_p = torch.log(prior_pi + 1e-6)
            
            kl_z = torch.sum(q*(log_q - log_p))
            kl_sticks = torch.sum(kl_divergence_kumaraswamy(torch.ones_like(a_soft), a_soft, b_soft))
            self.n += len(self.t_pi.view(-1))
            self.n += len(a_soft.view(-1))

        return z, kl_sticks, kl_z


    def lwta_activation(self, input, temp = 0.67, hard = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.

        logits = torch.reshape(input, [-1,input.size(-2), self.K, self.U])
        rand=True
        factor=0.2
        if not self.training:
            rand=False
            factor=1
            
        xi = concrete_sample(logits, temperature = temp, hard = hard,rand=rand)
        out = logits*xi
      
        out = out.reshape(input.shape)
       
        if False : 
            logits=torch.unsqueeze(input,-1)
            r1=torch.roll(torch.roll(logits,-3,-2),8,-3)
            r2=torch.roll(torch.roll(logits,-1,-2),1,-3)
            r3=torch.roll(torch.roll(logits,1,-2),-1,-3)
            r4=torch.roll(torch.roll(logits,3,-2),-8,-3)
            lr= torch.cat((logits,logits),-1)
            lr[:,:,::4,1]=r1[:,:,::4,0]
            lr[:,:,1::4,1]=r2[:,:,1::4,0]
            lr[:,:,2::4,1]=r3[:,:,2::4,0]
            lr[:,:,3::4,1]=r4[:,:,3::4,0]
            n=0
            if self.training:
                n=torch.normal(lr*0,0.5)
            xi = concrete_sample(lr+n, temperature = temp, hard = hard)
            out = lr*xi
            out=out[:,:,:,0]
            
            
        if False : 
            logits=torch.unsqueeze(input,-1)
            r1=torch.roll(torch.roll(logits,1,-2),1,-3)
            r2=torch.roll(torch.roll(logits,-1,-2),-1,-3)
            
            lr= torch.cat((logits,logits),-1)
            lr[:,:,::2,1]=r1[:,:,::2,0]
            lr[:,:,1::2,1]=r2[:,:,1::2,0]
            n=0
            if self.training:
                n=torch.normal(lr*0,0.1)
            xi = concrete_sample(lr+n, temperature = temp, hard = hard)
            out = lr*xi
            out=out[:,:,:,0]
            logits=lr
        
        
        if self.training:
            q = F.softmax(logits, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
        return out, kl*100

    def hiddenlwta(self,input,shift,temp,hard):
        kl = 0.
        
        
        logits = input*1.0
        
        logits = torch.roll(logits,shift,-1)
        
        if shift<0:
            logits = torch.reshape(logits, [-1,input.size(-2),input.size(-1), 1])
            logits  = torch.cat([ logits[:,:,i::self.U,:] for i in range(self.U)],axis=-2)
            logits = torch.reshape(logits, [-1,input.size(-2), self.K, self.U])
        else:
            logits = torch.roll(logits,shift,-1)
            logits = torch.reshape(logits, [-1,input.size(-2), self.K, self.U])
        xi = concrete_sample(logits, temperature = temp, hard = hard,rand=True)
       
       
        if self.training:
            q = F.softmax(logits, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
            
       
            
        if shift<0:
            xi=xi.reshape(input.shape)
            xo=xi*0
            
            for i in range(self.U):
                xo[:,:,i::self.U]=xi[:,:,self.K*i:self.K*(i+1)]
            xi=xo
            
        else:
        
            xi=torch.roll(xi.reshape(input.shape),-shift,-1) 
            
        
        
        return xi, kl*100    
    
    
    def superlwta_activation(self, input, temp = 0.67, hard = False):
       
        kl = 0.

       
        
      
            
        x1,kl1 = self.hiddenlwta(input,0,temp,hard)
        x2,kl2 = self.hiddenlwta(input,-1,temp,hard)
        
        if self.training :
            p=4
         
        else : 
            p=16
      #  xi=T*torch.log(torch.exp((x1)/T)+torch.exp((x2)/T))-0.693*torch.rand(x1.shape).cuda()*T
        xi=torch.pow(torch.pow(x1+0.001,p)+torch.pow(x2+0.001,p),1/p)
        
       # print(xi[0,0])
           

        # xi  = T*torch.log(torch.exp(x1*r1/T)+torch.exp(x2*r2/T)) -torch.log(r1+r2)*T*0.5
        kl=kl+kl1+kl2
        if not self.training and np.random.uniform()<0.001:
            print(torch.mean(xi))

            
        out = input*xi
        
        #out = out.reshape(input.shape)
        if np.random.uniform()<0.00001:
            print()
            print(xi[0,0,:8])
            print(out[0,0,:8])
            print()
        return out, kl    
    
    def new_activation(self, input, temp = 0.67, hard = False):
       

        kl = 0.

        logits = torch.reshape(input, [-1,input.size(-2), self.K, self.U])
        mean = torch.mean(torch.exp(logits),-1,keepdim=True)
        if self.training:
            temp=0.67
        xi = bin_concrete_sample(torch.exp(logits)-(mean), temperature = temp*5, hard = hard)
        xi=xi*0.99+0.01
        out = logits*xi
        if  self.training and np.random.uniform()<0.005:
            
            M=xi.to('cpu').detach().numpy()
            N=logits.to('cpu').detach().numpy()
            M=M[0,0,0:5,:]
            N=N[0,0,0:5,:]
           
            print(pd.DataFrame(M).round(2))
            print(pd.DataFrame(N).round(2))
        out = out.reshape(input.shape)
        if self.training:
            q = F.sigmoid(torch.exp(logits)-(mean))
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(0.5))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
        return out, kl*100
        
        
    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return "input_features = {}, output_features = {}, bias = {}".format(
            self.input_features, self.output_features, self.bias
        )



###########################################################################
##################### BAYESIAN CONV IMPLEMENTATION ########################
###########################################################################
class ConvBayesian(Module):
    """
    Class for a Bayesian Conv Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, competitors,
                 activation, deterministic = True, temperature = 0.67, ibp = False, bias=True):
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

        self.K = out_channels // competitors
        self.U = competitors
        self.activation = activation
        self.deterministic = deterministic

        self.temperature = temperature
        self.ibp = ibp
        self.bias  = bias
        self.training = True
        self.tau = 1e-2

        if activation == 'lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2')
                self.U = 2
                self.K = out_channels // self.U
            if out_channels % competitors != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} feature maps in groups of {}..'.format(out_channels, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.U = 1
                self.K = out_channels // self.U

        self.posterior_mean = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                     self.kernel_size, self.kernel_size))

        if not deterministic:
            # posterior unnormalized scale. Needs to be passed by softplus
            self.posterior_un_scale = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                         self.kernel_size, self.kernel_size))
            self.register_buffer('weight_eps', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_channels))
            if not self.deterministic:
                self.bias_un_scale = Parameter(torch.Tensor(out_channels))
                self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            if not self.deterministic:
                self.register_parameter('bias_un_scale', None)
                self.register_buffer('bias_eps', None)

        if ibp:
            self.conc1 = Parameter(torch.Tensor(self.K))
            self.conc0 = Parameter(torch.Tensor(self.K))
            self.t_pi = Parameter(torch.Tensor(self.K))

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
        init.xavier_normal_(self.posterior_mean)

        if not self.deterministic:
            self.posterior_un_scale.data.fill_(-5.)

        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.posterior_mean)
            bound = 1. / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound, bound)


            if not self.deterministic:
                self.bias_mean.data.fill_(0.0)
                #self.bias_un_scale.data.fill_(-5.)

        if self.ibp:
            self.conc1.data.fill_(2.)
            self.conc0.data.fill_(0.5453)

            init.uniform_(self.t_pi, .1, .1)


    def forward(self, input):
        """
        Overrride the forward pass to match the necessary computations.

        :param input: torch tensor: the input to the layer.

        :return: torch tensor: the output of the layer after activation
        """
        layer_loss = 0.

        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = torch.nn.Softplus(self.posterior_un_scale)
                W = self.posterior_mean + posterior_scale * torch.randn_like(posterior_scale)
                kl_weights = -0.5 * torch.mean(2 * posterior_scale - torch.square(self.posterior_mean)
                                               - posterior_scale ** 2 + 1)
                layer_loss += torch.sum(kl_weights)
            else:
                W = self.posterior_mean

            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z*W

                layer_loss += kl_sticks
                layer_loss += kl_z


            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale) * torch.randn_like(self.bias_un_scale)
                else:
                    bias = self.bias_mean
            else:
                bias = None


        else:
            W = self.posterior_mean
            bias = self.bias_mean

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.01)
                W = z*W

        out = F.conv2d(input, W, bias, stride = self.stride, padding = self.padding)

        if self.activation == 'linear':
            self.loss = layer_loss
            return out

        elif self.activation == 'relu':
            self.loss = layer_loss
            return F.relu(out)

        elif self.activation == 'lwta':
            out, kl = self.lwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            return out

        else:
            raise ValueError(self.activation + " is not implemented..")


    def indian_buffet_process(self, temp =0.67):

        kl_sticks = kl_z = 0.
        z_sample = bin_concrete_sample(self.t_pi, temp)

        if not self.training:
            t_pi_sigmoid = torch.sigmoid(self.t_pi)
            mask = t_pi_sigmoid > self.tau
            z_sample = mask*t_pi_sigmoid

        z = z_sample.repeat(self.U)

        if self.training:
            a_soft = F.softplus(self.conc1)
            b_soft = F.softplus(self.conc0)

            q_u = kumaraswamy_sample(a_soft, b_soft, sample_shape = [a_soft.size(0)])
            prior_pi = torch.cumprod(q_u, -1)

            q = torch.sigmoid(self.t_pi)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(prior_pi + 1e-8)

            kl_z = torch.sum(q*(log_q - log_p))
            kl_sticks = torch.sum(kl_divergence_kumaraswamy(torch.ones_like(a_soft), a_soft, b_soft))

        return z[:, None, None, None], kl_sticks, kl_z



    def lwta_activation(self, input, temp):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.

        logits = torch.reshape(input, [-1, self.K, self.U, input.size(-2), input.size(-1)])
        rand=True
        if not self.training:
            rand=False
            
        xi = concrete_sample(logits, temp, hard = False,rand=rand)

        out = logits * xi
        out = torch.reshape(out, input.shape)

        if self.training:
            q = F.softmax(logits, dim =2)
            log_q = torch.log(q + 1e-8)
            kl = torch.mean(q*(log_q - torch.log(torch.tensor(1.0/ self.U))), 0)
            kl = torch.sum(kl)


        return out, kl


    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return "prior_mean = {}, prior_scale = {}, input_features = {}, output_features = {}, bias = {}".format(
            self.prior_mean, self.prior_scale, self.in_channels, self.out_channels, self.bias
        )
