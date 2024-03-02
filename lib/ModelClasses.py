import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

    
class MaskedHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd,
                                         block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
    
    
class MaskedMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size, n_embd,
                                         block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, mask=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        if mask:
            self.sa = MaskedMultiHeadAttention(n_embd, n_head, head_size, block_size)
        else:
            self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
            
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers, n_head, device):
        super().__init__()
                             
        self.device = device
                     
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,
                                            n_head=n_head, mask=True,
                                            block_size=block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None, device="cpu"):
        B, T = index.shape
        
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens, block_size, device):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond, device=device)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
    
    
class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers,
                 n_head, device, 
                 conv_features=[64, 16, 8, 2],
                 linear_features=[128, 32, 16, 8]):
        super().__init__()

        self.device = device
                     
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(
                                    n_embd, 
                                    n_head=n_head,
                                    block_size=block_size) for _ in range(n_layers)]
                                   )
        self.ln_f = nn.LayerNorm(n_embd)
        
        layers = []
        start = 1
        for f in conv_features:
            layers.append(self.ConvBlock(start, f))
            start = f
        self.conv = nn.Sequential(*layers)
        
        layers = []
        start = n_embd
        for f in linear_features:
            layers.append(self.DenseBlock(start, f))
            start = f
        self.linear = nn.Sequential(*layers)
        
        self.final = nn.Linear(linear_features[-1], 3)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
    def ConvBlock(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
    
    
    def DenseBlock(self, inp, out, final=True):
        if final:
            return nn.Linear(inp, out)
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.BatchNorm1d(out),
            nn.ELU(),
            nn.Dropout(0.2)
        )
            
            
    def forward(self, index, targets=None):
        B, T = index.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        x = x.unsqueeze(1)
        
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.final(x)
        
        return x

    
class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, inp, layers = [1024, 256, 64, 32, 8]):
        super().__init__()
        self.latent_dims = latent_dims
        self.layers = layers
        self.inp = inp
        
        self.initial_enc = self.dense_block(inp, layers[0])
        
        layer = []
        start = layers[0]
        for l in layers[1:]:
            layer.append(self.dense_block(start, l))
            start = l
        self.encoder = nn.Sequential(*layer)
        
        self.latent_layer = self.dense_block(layers[-1], latent_dims, final=True)
        
        self.initial_dec = self.dense_block(latent_dims, layers[-1], final=True)
        
        layer = []
        start = layers[-1]
        for l in layers[::-1][1:]:
            layer.append(self.dense_block(start, l))
            start = l
        self.decoder = nn.Sequential(*layer)
        
        self.final = self.dense_block(layers[0], inp, final=True)
        
    def dense_block(self, inp, out, final=False):
        if final:
            return nn.Linear(inp, out)
        return nn.Sequential( 
                nn.Linear(inp, out),
                nn.BatchNorm1d(out),
                nn.ELU(inplace=0.2))
    
    def encode(self, X):
        x = self.initial_enc(X)
        x = self.encoder(x)
        x = self.latent_layer(x)

        return x
    

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std
    
    
    def decode(self, X):
        x = self.initial_dec(X)
        x = self.decoder(x)
        x = self.final(x)
        return x
    
    
    def forward(self, X):
        x = self.encode(X)
        x = self.decode(x)
        return x


    def fit(self, train_data, val_data, optimizer, loss_fn, epochs, device):
        losses = {"train_loss":[], "val_loss":[]}

        iter = trange(epochs)
        iter.set_postfix_str(f"train_loss=-----, val_loss=-----")

        for i in iter:
            train_loss, val_loss = [], []
            self.train()
            for X, y in train_data:
                X = X.to(device)

                X_pred = self.forward(X)
                loss = loss_fn(X_pred, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach().cpu().item())

            self.eval()
            with torch.no_grad():
                for X, y in val_data:
                    X = X.to(device)
    
                    X_pred = self.forward(X)
                    loss = loss_fn(X_pred, X)
                    val_loss.append(loss.detach().cpu().item())

            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            
            losses["train_loss"].append(train_loss)
            losses["val_loss"].append(val_loss)
            iter.set_postfix_str(f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        return losses


class EmbedLoader:
    def __init__(self, embds, labels):
        self.embds = embds
        self.labels = labels
        self.len = len(embds)
        self.shape = embds[0].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.embds[i], self.labels[i]

    def __repr__(self):
        return "a dataloader for fetching labelled embeddings"