#Transformer arcitecure
#  the transformer block is repeated 12 times in gpt-2
#  1. Multi-head Attention: We look at input X, and multi the X with trainable weight matrix Query(q), Key(K), Value(v)
#        will take the dot produt of the of the query and Key which procide the Attention scores and attention score are normailised to provide the attention weights
#        attention weithgt are multiply with the value matrix which provide the context vectors(it not only capture semantic meaning of that particluar word, but also capture how the relationship of that word relate to other words of sentence or how much attention that particular worlds give attnetion to the other words)
#        - these are richer representation than the embedding vector (it contains semantic meaning of particular vector -  it contains more info how that particular word relate to the other words in a senctence)
#  2. Lyaert normalisation = Normalisaing the layer outputs, let save any layer puput mean has = 0.12 , variance - 0.39 , after applyin gthe layer normat ( ) . it lieads to a bit of the stability during  backprogration it ensure that that value sare too large so that the gradiesnt doesn;'t vanish 2. it solves th aprob of internal co-variance  shitft ( durong the traingn process the input recive in certian layer may have difference distrubution bcz that halts the trainging )
        # it implement before the multi - head attention and before the feed forward
#   3. Dropout layer can be apply to any layer - it loks at the layer otuput and randomly turns out of some of the outputs. why ? it improve generalisaion- dutin traning , the neiron get lazy dont update themselves, it provents overfitting
#   4. Feed forward: it preserve the dimension of the input -  1. layer expalnsion which means that a hidden layer is used ( which is 4 times larger than the embedding dimension) 2. compression - comeback to the same dimension ( why ? - to explpre richer place of parameters - expansipn - it takes input inot much higher 4 times the dimension - where it will uncove much detail parmeters and eventulaly llm learn better every neuron has an activation functin which uses the GELU activation dunction - differenta ar x =0
# )
#   5. Shortcut connection : the output of 1 layer is added with the outpur of the previous layer.  it solves the vanishing gradient problems - if the gradient becomes very small the learning stops

# when a transformer block process an input seq, each element is represent by a fixed size vector
# the operations within the transformer block such as multi-head attention and feed forward layers, are designerd to transform thr input vectods such that their demisionlau is presented
# self attention ---> analyse the relation between input elements.
#  feed forward network ---> modifies data indivially at each position
# 
import torch
import torch.nn as nn

from model import MultiHeadAttention


GPT_CONFIG_124M={
    "vocab_size" : 50257,
    "context_length" : 4,
    "emb_dim": 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False  
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-8
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward( self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim =-1 , keepdim= True, unbiased = False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift
    

class GELU(nn.Module):
    def ___init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0 /torch.pi)) * (x + 0.044715) * torch.pow(x,3)))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features = cfg["emb_dim"], out_features = 4* cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features =  4 * cfg["emb_dim"], out_features = cfg['emb_dim']),
              
        )
    def forward(self,x):
        return self.layers(x)
    
    



class TransformerBlock(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(embed_dim= cfg["emb_dim"], 
                                      out_dim = cfg['emb_dim'],
                                      seq_len =cfg["context_length"],
                                      num_heads = cfg['n_heads'],
                                      dropout = cfg["drop_rate"],
                                      qvk_bias = cfg["qkv_bias"]
                                      )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])
                
    
    # shortcut connection for attention block
    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        print(type(self.norm1))  
        x = self.att(x) 
        x = self.dropout_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut
        return x
    
    
torch.manual_seed(123)
x = torch.rand(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print(output)
print("Input shape:", x.shape)
print("Output shape:", output.shape)