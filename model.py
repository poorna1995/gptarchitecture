import torch
import torch.nn as nn
import tiktoken
import math


class InputEmbedding(nn.Module):
    def __init__(self,text, vocab_size, embed_dim, dropout):
        super().__init__()
    # intialisation for inputembedding ( tokenisation, token embedding, poistional embedding)
        
        self.vocab_size  = vocab_size
        self.embedding_layer  = nn.Embedding(vocab_size,embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    # used inbuilt tiktoken library to split the text into unque token id

        tokenisor = tiktoken.get_encoding("cl100k_base")
        self.token_id  = torch.tensor(tokenisor.encode(text), dtype=torch.long)
        self.seq_len = len(self.token_id)
        print(self.token_id)
       
        
        
    # positional embedding intialisation
        self.embed_dim = embed_dim
    
    # create an tensor with len of the tokens and dimensi
        pe = torch.zeros(self.seq_len, self.embed_dim) # (4,768)
        
  # tensor representing the positions of tokens in the input sequence.
        position = torch.arange(0, self.seq_len, dtype = torch.float).unsqueeze(1) # (4,1)
        
   #calculating the values that will be used in the positional encoding of the input embeddings.
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim
                                                                           ))

    # calculating the sine positional encoding for even indices in the positional embedding tensor `pe`.
        pe[:, 0::2] = torch.sin(position * div_term) # silce all the rows and silce the even colums
        
     # calculating the sine positional encoding for even indices in the positional embedding tensor `pe`.    
        pe[:, 1::2] = torch.cos(position * div_term)
        
    #
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe',pe)
        
    def forward(self):
        embedding_vect = self.embedding_layer(self.token_id).unsqueeze(0)# 7 ---->(7,4)
        print(embedding_vect.shape)
        print(self.pe.shape)
        x = embedding_vect + self.pe[:, :embedding_vect.shape[1], :].requires_grad_(False)
        x = self.dropout(x)
        return x




class LayerNormalization(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim =True) # (1,7,6)
        variance = x.var(dim =-1, keepdim =True, unbiased = False)# (1,7,6)
        layer_nom = (x - mean) / torch.sqrt(variance + self.eps ) # (1,7,6)
        return (self.scale * layer_nom) + self.shift# (1,7,6)
    


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, out_dim ,seq_len,dropout, qvk_bias = False):
        super().__init__()
        self.out_dim   = out_dim
     
        self.w_q = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim) #(768,768)
        
        self.w_k = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        print(self.w_q)
        self.w_v = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        print(self.w_q)

    def forward(self,x):
        batch_size, seq_len, embed_dim = x.shape # ( 1, 4, 768) @ (1, 768,10) --? ( 1,4,10)
            
        query = self.w_q(x) # (batch_size , seq_len, embed_dim) @(batch_size , embed_dim, out_dim) --->  (batch_size , seq_len, out_dim)
        key   = self.w_k(x) # (batch_size , seq_len, embed_dim) @(batch_size , embed_dim, out_dim) --->  (batch_size , seq_len, out_dim)
        value = self.w_v(x) # (batch_size , seq_len, embed_dim) @(batch_size , embed_dim, out_dim) --->  (batch_size , seq_len, out_dim)
    
        attn_score = query @ key.transpose(1,2)            #(batch_size , seq_len, out_dim) @(batch_size , out_dim, seq_len) ----> (batch_size, seq_len, seq_len ) #  ( 1, 4,10) ( 1, 4,10) -->(1,4,10) (1,10,4) --> (1,4,4)
        scaled_up  = attn_score/ math.sqrt(out_dim)
        attn_weight = torch.softmax(scaled_up, dim =-1)  
        contxt_vec = attn_weight@value                     #  (batch_size, seq_len, seq_len ) @ (batch_size , seq_len, out_dim)  ---> (batch_size, seq_len, out_dim )
            
        return contxt_vec





class CasualAttention(nn.Module):
    def __init__(self, embed_dim, out_dim ,seq_len,dropout, qvk_bias = False):
        super().__init__()
        self.out_dim   = out_dim

        self.w_q = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        self.w_k = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        self.w_v = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1))
        
    def forward(self,x):
        query = self.w_q(x)  # (batch_size , seq_len, embed_dim) @(batch_size , embed_dim, out_dim) --->  (batch_size , seq_len, out_dim)
        key   = self.w_q(x)
        value = self.w_q(x)
        
        attn_score = query @ key.transpose(1,2)  ##(batch_size , seq_len, out_dim) @(batch_size , out_dim, seq_len) ----> (batch_size, seq_len, seq_len )
        
        attn_score.masked_fill_( 
            self.mask.bool(), -torch.inf)
        attn_weight = torch.softmax (attn_score/math.sqrt(out_dim) ,dim=-1)
        context_vec = attn_weight @ value
        return context_vec
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, out_dim ,seq_len,dropout, num_heads, qvk_bias = False):
        super().__init__()
        self.out_dim   = out_dim
        self.num_heads = num_heads
        self.head_dim = self.out_dim // self.num_heads
        
        self.w_q = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        self.w_k = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        self.w_v = nn.Linear(in_features = embed_dim, out_features = out_dim, bias = qvk_bias) # (batch_size, embed_dim, outdim)
        
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        
      
        
        
    def forward(self,x):
        batch_size, seq_len, embed_dim = x.shape

        query = self.w_q(x)  # (batch_size , seq_len, embed_dim) @(batch_size , embed_dim, out_dim) --->  (batch_size ,seq_len, out_dim)
        key   = self.w_q(x)
        value = self.w_q(x)
        
        
        
        query = query.view(batch_size, seq_len,self.num_heads, self.head_dim) #(batch_size ,seq_len,n um_heads, head_dim)
        key   = key.view(batch_size, seq_len,self.num_heads, self.head_dim) #(batch_size ,seq_len,n um_heads, head_dim)
        value = value.view(batch_size, seq_len,self.num_heads, self.head_dim) #(batch_size ,seq_len,n um_heads, head_dim)
        
        
      
        #  group by heads (batch_size ,seq_len,n um_heads, head_dim) @ (batch_size ,seq_len,num_heads, head_dim)
        
        key = key.transpose(1,2)     # (batch_size ,num_heads, seq_len head_dim)
        query = query.transpose(1,2) # (batch_size ,num_heads, seq_len head_dim)
        value = value.transpose(1,2) # (batch_size ,num_heads, seq_len head_dim)
        
        
        
        attn_score = query @ key.transpose(2,3)  ##batch_size ,num_heads, seq_len head_dim) @batch_size ,num_heads, seq_len head_dim)----> (batch_size,num_heads,seq_len, seq_len )

        attn_score.masked_fill_( 
            self.mask.bool(), -torch.inf)  ##batch_size ,num_heads, seq_len head_dim) @batch_size ,num_heads, seq_len head_dim)----> (batch_size,num_heads,seq_len, seq_len )
        
        # applt softwax for the intrepebility and Scaling, Improving Generalization
        attn_weight = torch.softmax (attn_score/math.sqrt(out_dim) ,dim=-1) #batch_size ,num_heads, seq_len head_dim) @batch_size ,num_heads, seq_len head_dim)----> (batch_size,num_heads,seq_len, seq_len )
        
        # apply dropout for Preventing Overfitting
        attn_weight = self.dropout(attn_weight)
    
        context_vec = (attn_weight @ value)    # (batch_size,num_heads,seq_len, seq_len ) @ (batch_size ,num_heads, seq_len head_dim) ---->(batch_size ,num_heads, seq_len, head_dim)
        context_vec = context_vec.transpose(1,2) #(batch_size ,seq_len, num_heads, head_dim)
        
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size,seq_len,self.out_dim)
        return context_vec




class FeedForward(nn.Module):
    def __init__(self,embed_dim):
       self.layers =  nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = 4*embed_dim),
            GELU(),
            nn.Linear(in_features = 4*embed_dim, out_features = embed_dim),
        )
    def forward(self,x):
        return self.layers(x)











   
text           = "how are you doing! this is" 
vocab_size     = 50267
embed_dim  = 4
dropout  = 0.1
out_dim  = 6

# The `InputEmbedding` class is responsible for processing the input text data. Here is what it does:
input_embedding = InputEmbedding(text, vocab_size, embed_dim, dropout)

input_embedding_output = input_embedding()
print(f"Input Embedding Output : {input_embedding_output.shape}")
print(f"Input Embedding Output : {input_embedding_output}")


layer_normalization= LayerNormalization(embed_dim)
output_layer_normalization = layer_normalization(input_embedding_output)
print(f"output_layer_normalization Output : {output_layer_normalization}")
print(output_layer_normalization.mean(dim=-1,keepdim =True))
print(output_layer_normalization.var(dim=-1,keepdim =True,unbiased = False))


seq_len = output_layer_normalization.shape[1]
self_attention = SelfAttention(embed_dim, out_dim ,seq_len,dropout, qvk_bias = False)
self_attention_output = self_attention(output_layer_normalization)
print(f"Self Attention Output : {self_attention_output.shape}")



seq_len = output_layer_normalization.shape[1]
casual_attention = CasualAttention(embed_dim, out_dim ,seq_len,dropout, qvk_bias = False)
casual_attention_output = casual_attention(output_layer_normalization)
print(f"Casual Attention Output : {casual_attention_output}")
print(f"Casual Attention Output Shape: {casual_attention_output.shape}")



seq_len = output_layer_normalization.shape[1]

multi_head_attention = MultiHeadAttention(embed_dim, out_dim ,seq_len,dropout, num_heads =2, qvk_bias = False)
multi_attention_output = multi_head_attention(output_layer_normalization)
print(f"Casual Attention Output : {multi_attention_output}")
print(f"Casual Attention Output Shape: {multi_attention_output.shape}")





# test with 2 bacthes

batch = torch.stack((input_embedding_output, input_embedding_output), dim=0).squeeze(1)
print(batch.shape)

batch_size, seq_len, embed_dim = batch.shape






multi_head_attention = MultiHeadAttention(embed_dim, out_dim ,seq_len,dropout, num_heads =2, qvk_bias = False)
multi_attention_output = multi_head_attention(batch)
print(f"multi_head_attention  Output : {multi_attention_output}")
print(f"multi_head_attention Output Shape: {multi_attention_output.shape}")




