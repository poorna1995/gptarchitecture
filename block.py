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
# from model import MultiHeadAttention


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 326,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, out_dim, context_length, dropout, num_heads, qvk_bias=False):
        super().__init__()
        assert (out_dim % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(embed_dim, out_dim, bias=qvk_bias)
        self.W_key = nn.Linear(embed_dim, out_dim, bias=qvk_bias)
        self.W_value = nn.Linear(embed_dim, out_dim, bias=qvk_bias)
        self.out_proj = nn.Linear(embed_dim, out_dim)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, embed_dim = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.out_dim)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


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
            context_length=cfg["context_length"],
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




import os
import urllib.request

file_path ="the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    with open(file_path, "w", encoding ="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding = "utf-8") as file:
        text_data = file.read()
        

# print(text_data)

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)


# With 5,145 tokens, the text is very short for training an LLM, but again, it's for educational purposes (we will also load pretrained weights later).

# Next, we divide the dataset into a training and a validation set and use the data loaders from chapter 2 to prepare the batches for LLM training.
    
# Since we train the LLM to predict the next word in the text, the targets look the same as these inputs, except that the targets are shifted by one position    



from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids =[]
        self.target_ids =[]
        
        # tokenise the entire text
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        
        # use a sliding windoe to chunk the text into overlapping sequences of max_lenght
        
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids [i:i+max_length]
            target_chunk = token_ids [i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size =4,max_length =256, stride = 128, shuffle=True,drop_last = True, num_workers =0):
    #intilaise the tokeniser
    tokenizer = tiktoken.get_encoding("gpt2")
    
    #create dataset
    dataset = GPTDatasetV1(
        txt, tokenizer, max_length, stride
        
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    return dataloader



train_ratio =0.90
split_idx = int( train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]



torch.manual_seed(123)


train_loader = create_dataloader(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
    
)


val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)




print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

print(len(train_loader))
print(len(val_loader))



train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)





class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference




def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)