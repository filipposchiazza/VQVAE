
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_activations import Swish


class ResBlock(nn.Module):

    def __init__(self, input_channels, groups=16):
        """Single Residual Block.

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        groups : int
            Number of groups for the GroupNorm layers.
        """
        super(ResBlock, self).__init__()
        self.channels = input_channels
        self.groups = groups
        self.resblock = nn.Sequential(nn.GroupNorm(num_groups=groups, num_channels=self.channels),
                                      Swish(),
                                      nn.Conv2d(in_channels=self.channels, 
                                                out_channels=self.channels, 
                                                kernel_size=3,
                                                padding=1),
                                      nn.GroupNorm(num_groups=groups, num_channels=self.channels),
                                      Swish(),
                                      nn.Conv2d(in_channels=self.channels, 
                                                out_channels=self.channels, 
                                                kernel_size=3,
                                                padding=1))
        
    def forward(self, x):
        return x + self.resblock(x)



class ResStack(nn.Module):
    "Residual blocks groupded together"
    def __init__(self, input_channels, num_resblock, groups=16):
        """Residual blocks groupded together
        
        Parameters
        ----------
        input_channels : int
            Number of input channels.
        num_resblock : int
            Number of Residual blocks to apply to each resolution.
        groups : int
            Number of groups for the GroupNorm layers.
        """
        super(ResStack, self).__init__()
        self.channels = input_channels
        self.num_resblock = num_resblock
        self.groups = groups
        self.resblock = nn.Sequential(*[ResBlock(input_channels=input_channels, groups=groups) for _ in range(num_resblock)])
        
    def forward(self, x):
        return self.resblock(x)



class DownSampleBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        """Downsampling block

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        """
        super(DownSampleBlock, self).__init__()
        self.input_ch = input_channels
        self.output_ch = output_channels
        self.downsample = nn.Conv2d(in_channels=self.input_ch, 
                                    out_channels=self.output_ch, 
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        
    def forward(self, x):
        return self.downsample(x)



class UpSampleBlock(nn.Module):
    
        def __init__(self, input_channels, output_channels):
            """Upsampling block
    
            Parameters
            ----------
            input_channels : int
                Number of input channels.
            output_channels : int
                Number of output channels.
            """
            super(UpSampleBlock, self).__init__()
            self.input_ch = input_channels
            self.output_ch = output_channels
            self.upsample = nn.ConvTranspose2d(in_channels=self.input_ch, 
                                               out_channels=self.output_ch, 
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1)
            
        def forward(self, x):
            return self.upsample(x)



class AttentionBlock(nn.Module):
        
    def __init__(self, input_channels, groups=16):
        """Attention block

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        groups : int
            Number of groups for the GroupNorm layers.
        """
        super(AttentionBlock, self).__init__()
        self.channels = input_channels
        self.groups = groups
        self.group_norm = nn.GroupNorm(num_groups=groups, 
                                       num_channels=self.channels)
        self.query = nn.Conv2d(in_channels=self.channels, 
                               out_channels=self.channels, 
                               kernel_size=1,
                               padding=0)
        self.key = nn.Conv2d(in_channels=self.channels, 
                             out_channels=self.channels, 
                             kernel_size=1,
                             padding=0)
        self.value = nn.Conv2d(in_channels=self.channels, 
                               out_channels=self.channels, 
                               kernel_size=1,
                               padding=0)

    def forward(self, x):
        x_norm = self.group_norm(x)
        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)

        b, c, h, w = q.shape
        # reshape for matrix multiplication
        q = q.view(b, c, h*w).permute(0, 2, 1)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        # attention
        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn).reshape(b, c, h, w)

        return x + A


        
class Encoder(nn.Module):

    def __init__(self, input_channels, channels, latent_dim, has_attention, num_resblock, groups=16):
        """Encoder

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        channels : list
            Number of channels for each convolutional downsampling step. The first convolution is applied to 
            the whole image
        latent_dim : int
            Dimensionality of the embedding vectors (D in the original paper).
        has_attention : bool
            Whether to use attention blocks or not.
        num_resblock : int
            Number of Residual blocks to apply to each resolution.
        groups : int
            Number of groups for the GroupNorm layers.
        """
        super(Encoder, self).__init__()
        self.input_ch = input_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.has_attention = has_attention
        self.num_resblock = num_resblock
        self.groups = groups
        self.model = self._build_encoder()
        self.num_parameters = self._calculate_num_parameters() # (p_train, p_non_train)

    def forward(self, x):
        return self.model(x)
    
    
    def _calculate_num_parameters(self):
        "Evaluate the number of trainable and non-trainable model parameters"
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train
    

    def _build_encoder(self):
        "Helper function to build the encoder"
        
        layers = []
        # First convolution
        layers.append(nn.Conv2d(in_channels=self.input_ch, 
                                out_channels=self.channels[0], 
                                kernel_size=3,
                                padding=1))
        
        # m x {ResStack, Downsampling}
        for i in range(len(self.channels)-1):
            layers.append(ResStack(input_channels=self.channels[i], 
                                   num_resblock=self.num_resblock,
                                   groups=self.groups))
            layers.append(DownSampleBlock(input_channels=self.channels[i], 
                                          out_channels=self.channels[i+1]))
            
        # ResStack to the last dimension
        layers.append(ResStack(input_channels=self.channels[-1], 
                               num_resblock=self.num_resblock,
                               groups=self.groups))
        
        # Attention Block
        layers.append(AttentionBlock(input_channels=self.channels[-1], 
                                     groups=self.groups))
        
        # Last Residual Block
        layers.append(ResBlock(input_channels=self.channels[-1], 
                               groups=self.groups))
        
        # GroupNorm, Swish, Conv2D
        layers.append(nn.GroupNorm(num_groups=self.groups, num_channels=self.channels[-1]))
        layers.append(Swish())
        
        layers.append(nn.Conv2d(in_channels=self.channels[-1], 
                                out_channels=self.latent_dim, 
                                kernel_size=3,
                                padding=1))
        
        return nn.Sequential(*layers)



class Decoder(nn.Module):
    
    def __init__(self, output_channels, channels, latent_dim, has_attention, num_resblock, groups=16):
        super(Decoder, self).__init__()
        self.out_channels = output_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.has_attention = has_attention
        self.num_resblock = num_resblock
        self.groups = groups
        
        self.model = self._build_decoder()
        self.num_parameters = self._calculate_num_parameters() # (p_train, p_non_train)
        
    
    def forward(self, x):
        return self.model(x)
    
    
    def _calculate_num_parameters(self):
        "Evaluate the number of trainable and non-trainable model parameters"
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train
    
    
    def _build_decoder(self):
        
        layers = []
        
        # First convolution for channels update
        layers.append(nn.Conv2d(in_channels=self.latent_dim, 
                                out_channels=self.channels[0], 
                                kernel_size=3,
                                padding=1))
        
        # First Residual Block
        layers.append(ResBlock(input_channels=self.channels[0], groups=self.groups))
        
        # Attention Block
        layers.append(AttentionBlock(input_channels=self.channels[0], groups=self.groups))
        
        # m x {ResStack, Upsampling} 
        for i in range(len(self.channels)-1):
            layers.append(ResStack(input_channels=self.channels[i], 
                                   num_resblock=self.num_resblock,
                                   groups=self.groups))
            
            layers.append(UpSampleBlock(input_channels=self.channels[i], 
                                        out_channels=self.channels[i+1]))
            
        # GroupNorm, Swish, Conv2D
        layers.append(nn.GroupNorm(num_groups=self.groups, num_channels=self.channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels=self.channels[-1], 
                                out_channels=self.out_channels, 
                                kernel_size=3,
                                padding=1))

        return nn.Sequential(*layers)



class VectorQuantizer(nn.Module):
    
    def __init__(self, num_emb, emb_dim, beta):
        """ Vector Quantization Layer
        

        Parameters
        ----------
        num_emb : int
            Number of embedding vector used for the codebook (K in the original paper).
        emb_dim : int
            Dimensionality of the embedding vectors (D in the original paper).
        beta : float
            Weight term in the loss function.

        Returns
        -------
        None.

        """
        super(VectorQuantizer, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta
        
        self.emb = nn.Embedding(num_embeddings=num_emb, 
                                embedding_dim=emb_dim)
        self.emb.weight.data.uniform_(-1/self.num_emb, 1/self.num_emb)
        
    
    def forward(self, inputs):
        # convert inputs from BCHW to BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # flatten inputs
        flat_input = inputs.view(-1, self.emb_dim)
        # calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.emb.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.emb.weight.t()))
        
        # encoding
        encoding_indices= torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_emb, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # quantize and unflatten
        quantized = torch.matmul(encodings, self.emb.weight).view(input_shape)
        
        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)   
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # to copy-paste the gradient
        quantized = inputs + (quantized - inputs).detach()
        
        # evaluate the codebook usage
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices, perplexity
    
    
#%%
class VectorQuantizerEMA(nn.Module):
    
    def __init__(self, num_emb, emb_dim, beta, decay, epsilon=1e-5):
        """ Vector Quantization Layer with EMA coodebooks update 
        

        Parameters
        ----------
        num_emb : int
            Number of embedding vector used for the codebook (K in the original paper).
        emb_dim : int
            Dimensionality of the embedding vectors (D in the original paper).
        beta : float
            Weight term in the loss function.
        decay : float
            Weight in the EMA update

        Returns
        -------
        None.

        """
        super(VectorQuantizerEMA, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        
        self.emb = nn.Embedding(num_embeddings=num_emb, 
                                embedding_dim=emb_dim)
        self.emb.weight.data.normal_()
        
        self.register_buffer(name='_ema_cluster_size', 
                             tensor=torch.zeros(num_emb))
        self._ema_w = nn.Parameter(torch.Tensor(num_emb, emb_dim))
        self._ema_w.data.normal_()
        
    
    def forward(self, inputs):
        # convert inputs from BCHW to BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # flatten inputs
        flat_input = inputs.view(-1, self.emb_dim)
        # calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.emb.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.emb.weight.t()))
        
        # encoding
        encoding_indices= torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_emb, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # quantize and unflatten
        quantized = torch.matmul(encodings, self.emb.weight).view(input_shape)
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = (self.decay * self._ema_cluster_size 
                                      + (1 - self.decay) * torch.sum(encodings, 0))

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self.epsilon) 
                                      / (n + self.num_emb * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)

            self.emb.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))


        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.beta * e_latent_loss
        
        # to copy-paste the gradient
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # evaluate the codebook usage
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # extract encoding indices for later generation
        codes = encoding_indices.view(input_shape[:-1])
        
        return loss, quantized, encoding_indices, perplexity, codes



   
#%%

class BottleneckResBlock(nn.Module):
    "Bottleneck Residual block"
    def __init__(self, input_channels):
        super(BottleneckResBlock, self).__init__()
        self.input_ch = input_channels
        self.hidden_ch = int(input_channels/2)
        self.output_ch = input_channels
        
        self.conv1 = nn.Conv2d(in_channels=self.input_ch, 
                               out_channels=self.hidden_ch, 
                               kernel_size=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_ch, 
                               out_channels=self.hidden_ch, 
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=self.hidden_ch, 
                               out_channels=self.output_ch, 
                               kernel_size=1,
                               padding=0,
                               bias=False)
    
    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(res)
        res = self.conv2(res)
        res = F.relu(res)
        res = self.conv3(res)
        return x + res
    
    


#%%

class Encoder(nn.Module):
    
    def __init__(self, input_ch=3, 
                 channels = [64, 128, 256, 512],
                 num_resblock=2):
        
        """Convolutional Encoder
        

        Parameters
        ----------
        input_ch : int, optional
            Number of channels of the original images (3 for RGB, 1 for GrayLevel). The default is 3.
        channels : list of int
            Number of channels for each convolutional downsampling step. The first convolution is applied to 
            the whole image
        num_resblock : int, optional
            Number of Residual blocks to apply to each resolution. The default is 2.
            
        Returns
        -------
        None.

        """
        
        super(Encoder, self).__init__()
        self.input_ch = input_ch
        self.channels = channels
        self.num_resblock = num_resblock
        self.layers = self._build_encoder()
    
    
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
        
    
    def _calculate_num_parameters(self):
        "Evaluate the number of trainable and non-trainable model parameters"
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train
    
    
    def _build_encoder(self):
        "Helper function to build the encoder"
        
        layers = nn.ModuleList()
        
        # First convolution and residual block to the initial size image
        layers.append(nn.Conv2d(in_channels=self.input_ch, 
                                out_channels=self.channels[0], 
                                kernel_size=3,
                                padding='same',
                                bias=False))
        layers.append(ResStack(input_channels=self.channels[0], 
                               num_resblock=self.num_resblock))
        
        for i in range(len(self.channels)-1):
            layers.append(nn.Conv2d(in_channels=self.channels[i], 
                                    out_channels=self.channels[i+1], 
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False))
            layers.append(ResStack(input_channels=self.channels[i+1], 
                                   num_resblock=self.num_resblock))
        
        return layers
    
#%%

class Decoder(nn.Module):
    
    def __init__(self, input_ch,
                 output_ch,
                 channels = [512, 256, 128, 64],
                 num_resblock=2):
        
        """Convolutional decoder
        

        Parameters
        ----------
        input_ch : int
            Number of channels of the embedding representation.
            In other words, the number of output channels of the coupled encoder. 
        output_ch : int
            Number of channels of the reconstructed images (3 for RGB, 1 for GrayLevel).
        channels : list
            Number of channels for each convolutional upsampling step.
        num_resblock : int
            Number of Residual blocks to apply to each resolution. The default is 2.
            
        Returns
        -------
        None.

        """    
        super(Decoder, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.channels = channels
        self.num_resblock = num_resblock
        self.layers = self._build_decoder()
        
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
    
    
    def _calculate_num_parameters(self):
        "Evaluate the number of trainable and non-trainable model parameters"
        p_total = 0
        p_train = 0
        for p in self.parameters():
            p_total += p.numel()
            if p.requires_grad:
                p_train += p.numel()       
        p_non_train = p_total - p_train
        return p_train, p_non_train
    
    
    def _build_decoder(self):
        "Helper function to build the decoder"
        layers = nn.ModuleList()
        
        for i in range(len(self.channels)-1):
            layers.append(ResStack(input_channels=self.channels[i], 
                                   num_resblock=self.num_resblock))
            layers.append(nn.Conv2d(in_channels=self.channels[i], 
                                    out_channels=self.channels[i+1], 
                                    kernel_size=1,
                                    padding=0,
                                    bias=False))
            layers.append(nn.Upsample(scale_factor=2.0, mode='bilinear'))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=self.channels[-1], 
                                out_channels=self.output_ch, 
                                kernel_size=3,
                                padding=1,
                                bias=False))
            
        return layers
    