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



import torch
import torch.nn as nn
from model import MultiHeadAttention


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-8
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):  # Fixed the typo here
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg['emb_dim']),  # expansion
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg['emb_dim'])  # contraction
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            embed_dim=cfg["emb_dim"],
            out_dim=cfg['emb_dim'],
            seq_len=cfg["context_length"],
            num_heads=cfg['n_heads'],
            dropout=cfg["drop_rate"],
            qvk_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut
        return x


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg['vocab_size'], bias=False)

    def forward(self, input):
        batch_size, seq_len = input.shape
        tok_emb = self.tok_emb(input)
        pos_indices = torch.arange(seq_len, device=input.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_indices).expand(batch_size, -1, -1)
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    
def pad_sequence_to_context_length(sequence, context_length):
    # Pad sequence to match context length if needed
    padding_size = context_length - sequence.size(1)
    if padding_size > 0:
        padding = torch.zeros(sequence.size(0), padding_size, dtype=sequence.dtype, device=sequence.device)
        sequence = torch.cat((padding, sequence), dim=1)
    return sequence

def GenerateNextToken(model, inputs, max_token, context_length):
    for _ in range(max_token):
        # idx_cond = inputs[:, -context_length:]
        idx_cond = pad_sequence_to_context_length(inputs[:, -context_length:], context_length)
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        inputs = torch.cat((inputs, idx_next), dim=1)

    return inputs



# def TextToToken(text):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     encoded = tokenizer.encode(text, allowed_special ={'<|endoftext|'})
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)
#     return encoded_tensor

# def TokenToText(token_ids):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     flat = token_ids.
    
    
    

model = GptModel(GPT_CONFIG_124M)      

torch.manual_seed(123)
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
model = GptModel(GPT_CONFIG_124M)


start_context = "Everything everyone at once"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()
out = GenerateNextToken(
    model=model,
    inputs=encoded_tensor,
    max_token=10,
    context_length=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output:", out.shape)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
