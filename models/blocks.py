import torch
import torch.nn as nn
import torch.nn.functional as F

#This is actually a smaller version of dream_models.py 
#It contains the building blocks for dream_models.py 
#It has the same structure as it because we can imagine a model as a big block
#That contains other smaller blocks
#Both inherits nn.Module 
#This one is also full of smaller "Blocks", and those are the layers from nn like Conv1d or Maxpool

class FirstConvBlock(nn.Module):
    """
    Basic convolutional block.
    Consists of a convolutional layer, a max pooling layer and a dropout layer.
    """
    #Initializing variables like before
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        pool_size: int, 
        dropout: float
    ):
        # Initialize variables like before
        super().__init__()

        #Each first conv block consist of three layers
        # This layer is the convolution layer that reads the shit
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        # This layer is the pooling layer that down sizes sutff
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        # Drop out to deactive neurons randomly
        self.do = nn.Dropout(dropout)

    # Forward is basically what each layer do when data is passed to it 
    # In this case, the data go through all the layers and the processed one is returned
    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv(x))  # (batch_size, out_channels, seq_len)
        x = self.mp(x)  # (batch_size, out_channels, seq_len // pool_size)
        x = self.do(x)  # (batch_size, out_channels, seq_len // pool_size)
        return x
    
    #Squeeze and excite layer   
        #Basically squeeze each channel into one single number, then use two linear channels followed bytearray
        #an activation function to calculate the importance of each
        #Combine this with the previous layer
class SELayerSimple(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            reduction: int=4
        ):
        super().__init__()
        #This is the layer to excite
        self.fc = nn.Sequential(
                nn.Linear(out_channels, int(in_channels // reduction)),
                nn.SiLU(),
                nn.Linear(int(in_channels // reduction), out_channels),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        #Squeeze
        y = x.view(b, c, -1).mean(dim=2)
        #y are the weights
        y = self.fc(y).view(b, c, 1)
        #Return x*y so it combines importance with original values
        return x * y

class SwiGLULayer(nn.Module):
    #This is the swish layer
    def __init__(self, dim):
        super(SwiGLULayer, self).__init__()
        self.dim = dim
        self.swish = nn.SiLU() # same as swish

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.swish(gate)



class FeedForwardSwiGLU(nn.Module):
    #Apply two linaer transformations using embedding dim 
    #Prepares for th swiGLY layer
    def __init__(self, embedding_dim, mult=4, rate = 0.0, use_bias = True):
        super(FeedForwardSwiGLU, self).__init__()
        swiglu_out = int(embedding_dim * mult/2)
        self.layernorm = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.linear1 = nn.Linear(embedding_dim,embedding_dim * mult, bias = use_bias)
        self.swiglulayer = SwiGLULayer(dim = 1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out,embedding_dim, bias = use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1,2)) # Swap dimensions and make channel dim=2
        x = self.linear1(x) 
        x = self.swiglulayer(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        x = self.drop(x)
        x = self.linear2(x.transpose(1,2)) # Swap dimensions and make channel dim=2
        out = self.drop(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        return out

class ConformerSASwiGLULayer(nn.Module):
    def __init__(self, embedding_dim,  ff_mult = 4, kernel_size = 15, rate = 0.2, num_heads = 4, use_bias = False):
        super(ConformerSASwiGLULayer, self).__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same', bias = False),
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, bias = True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim,eps = 1e-6)    
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first = True)
        self.ff2 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        
        x1 = x.transpose(1,2)
        x1 = self.layernorm1(x1) #channel dim = 2
        x1 = x1.transpose(1, 2)
        x1 = x1 + self.conv(x1)
        
        x = x + x1
        x = x.transpose(1, 2) # output channel dim = 2
        x = self.layernorm2(x)
        x = x + self.attn(x, x, x)[0]
        x = x.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        
        return x

class YeastFinalBlock(nn.Module):
    def __init__(self,in_channels,out_channels=18):
        super().__init__()
        self.final_mapper = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding='same'
        )
    
    def forward(self, x):
        #applies a conv layer
        x = self.final_mapper(x)
        # 1 means only 1 output
        x = F.adaptive_avg_pool1d(x, 1)
        #Squeeze removes dimensions with size 2
        x = x.squeeze(2) 
        # log_softmax the result
        logprobs = F.log_softmax(x, dim=1)
        return logprobs

class HumanFinalBlock(nn.Module):
    def __init__(self,in_channels,out_channels=256):
        super().__init__()
        self.final_mapper = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding='same'
        )
        self.final_linear = nn.Linear(out_channels, 1) # for human
    
    def forward(self,x):
        x = self.final_mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        # For human we want to apply a linear final transform instead of a log one
        x = self.final_linear(x)
        return x
