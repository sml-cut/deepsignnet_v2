import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from signjoey.helpers import freeze_params
from signjoey.layers import  DenseBayesian,LWTA

####################################################

from torch.hub import load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = True, progress: bool = True, **kwargs) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model

###############################################################################################3



def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


        
        
        
class EmbeddingsEnsemble(nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
        N=10
        self.Decoders=[Embeddings_(*args, **kwargs) for i in range(N) ]
        self.Decoders=nn.ModuleList(self.Decoders)
        self.embedding_dim=self.Decoders[0].embedding_dim
        self.lut=self.Decoders[0].lut
        for param in self.Decoders[0].parameters():
            param.requires_grad = False
    def      forward(self,*args, **kwargs):
        return   [self.Decoders[i](*args, **kwargs) for i in range(10)]
# TODO (Cihan): Spatial and Word Embeddings are pretty much the same
#       We might as well convert them into a single module class.
#       Only difference is the lut vs linear layers.
class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        scale: bool = False,
        scale_factor: float = None,
        norm_type: str = None,
        activation_type: str = None,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        bayesian : bool = False,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        
        self.bayesian=bayesian    
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        if bayesian:
            self.ln = DenseBayesian( self.embedding_dim, self.embedding_dim, competitors = 2,
                                activation = activation_type, prior_mean=0, prior_scale=1. )
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type and not self.bayesian :
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """

        x = self.lut(x)
        if self.bayesian:
            x=self.ln(x)
        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type and not self.bayesian:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )

class SpatialEmbeddingsEnsemble(nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
        N=10
        
        self.Decoders=[SpatialEmbeddings_(*args, **kwargs) for i in range(N) ]
        self.Decoders=nn.ModuleList(self.Decoders)
        self.embedding_dim =self.Decoders[0].embedding_dim
        for param in self.Decoders[0].parameters():
            param.requires_grad = False
    def      forward(self,*args, **kwargs):
        return [self.Decoders[i](*args, **kwargs) for i in range(10)]
    
class SpatialEmbeddings(nn.Module):

    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        bayesian : bool = False,
        ibp : bool = False,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.bayesian = bayesian

        ###################################

        # Intialize Alexnet
        self.alexnet = alexnet()
        self.alexnet.classifier[6] = nn.Linear(4096, embedding_dim)

        #####################################

        if bayesian:
            self.ln = DenseBayesian(self.input_size, self.embedding_dim, competitors =2 ,
                                activation = activation_type, prior_mean=0, prior_scale=1. , kl_w=0.1, ibp = ibp)
         
        else:
            self.ln = nn.Linear(self.input_size, self.embedding_dim)
            #self.ln = nn.Conv1d(self.input_size,self.embedding_dim,5,padding=2)
          #  print('\n\n\n\n\n\n CONV  \n\n\n\n\n')
        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if bayesian:
            self.activation_type = False
        else:
            self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
            
        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)
       
        if freeze:
            freeze_params(self)


    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """

        frames_input = False

        if frames_input:
            ###################################

            temp_batch = x.shape[0]
            temp_frames = x.shape[1]
            # x = torch.ones([181, 3, 227, 227], device='cuda:0')
            x = x.reshape((temp_batch * temp_frames, 3, 227, 227))

            # Intialize Alexnet
            x = self.alexnet(x)
            x = x.reshape((temp_batch, temp_frames, self.embedding_dim))
            # self.ln = nn.Linear(self.input_size, self.embedding_dim)

            #####################################

        else:
            # x = self.ln(x.transpose(-1,-2)).transpose(-1,-2)
            # nn.Linear(self.input_size, self.embedding_dim)
            x = self.ln(x)
       
        if self.norm_type:
            x = self.norm(x, mask)
        
        if  self.activation_type and (not self.bayesian):
            x = self.activation(x)
        
        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )
