import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hides INFO/WARNING C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # silences the oneDNN notice

import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # compute output size of each head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

        self.out_proj = nn.Linear(d_out, d_out) # linear layer combines head outputs

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys       = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values   = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys       = keys.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values   = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (b, num_tokens, n_heads, head_dim)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # combine heads

        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # in case var is actually 0, add small eps to prevent division by 0

        # in case model performance improves without layer norm, 
        # scale and shift parameters will change significantly during training
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        # unbiased=False to keep computations consistent with original GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # arbitrary multiply by factor of 4
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x
        
class GPT(nn.Module):
    def __init__(self, cfg=GPT_CONFIG_124M, load=True, model_size='124M'):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        if load:
            model_dir = model_size
            tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
            print('GPT-2 checkpoint found!')
            settings = json.load(open(os.path.join(model_dir, "hparams.json")))
            params = self._load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
            print('Loading GPT-2 weights...',end="")
            self._load_weights_into_gpt(params)
            print('done!')

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits
    
    def _load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}

        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
            # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

            # Process the variable name to extract relevant parts
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            # Recursively access or create nested dictionaries
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            # Assign the variable array to the last key
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params
    
    def _load_weights_into_gpt(self, params):
        def assign(left, right):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                                "Right: {right.shape}"
                )
            return torch.nn.Parameter(torch.tensor(right))
        
        # set token and positional embedding weights
        self.pos_emb.weight = assign(self.pos_emb.weight, params['wpe'])
        self.tok_emb.weight = assign(self.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])): # for each transformer block

            # np.split divides weights into three equal parts for query, key, and value components
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            
            # Q, K, V weights
            self.trf_blocks[b].att.W_query.weight = assign(
                self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = assign(
                self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = assign(
                self.trf_blocks[b].att.W_value.weight, v_w.T)
            
            # Q, K, V biases
            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = assign(
                self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = assign(
                self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = assign(
                self.trf_blocks[b].att.W_value.bias, v_b)
            self.trf_blocks[b].att.out_proj.weight = assign(
                self.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.trf_blocks[b].att.out_proj.bias = assign(
                self.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])
            self.trf_blocks[b].ff.layers[0].weight = assign(
                self.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.trf_blocks[b].ff.layers[0].bias = assign(
                self.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.trf_blocks[b].ff.layers[2].weight = assign(
                self.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.trf_blocks[b].ff.layers[2].bias = assign(
                self.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])
            self.trf_blocks[b].norm1.scale = assign(
                self.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            self.trf_blocks[b].norm1.shift = assign(
                self.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            self.trf_blocks[b].norm2.scale = assign(
                self.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            self.trf_blocks[b].norm2.shift = assign(
                self.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])
            
        self.final_norm.scale = assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = assign(self.final_norm.shift, params["b"])

        # weight tying
        # re-use weights of the token embedding layer now in the output layer
        self.out_head.weight = assign(self.out_head.weight, params["wte"])

class GPT_SayThing(nn.Module):
    def __init__(self, cfg=GPT_CONFIG_124M, load=True, model_size='124M'):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # still needed for easy loading of GPT-2 pretrained weights
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        self.say_head = nn.Linear(
            cfg["emb_dim"], 33, bias=False
        )

        if load:
            model_dir = os.path.join('llm', 'gpt2', model_size)
            tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
            print('GPT-2 checkpoint found!')
            settings = json.load(open(os.path.join(model_dir, "hparams.json")))
            params = self._load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
            print('Loading GPT-2 weights...',end="")
            self._load_weights_into_gpt(params)
            print('done!')

            # freeze all layers
            for param in self.parameters():
                param.requires_grad = False

            # unfreeze last trf_block and layer norm
            for param in self.trf_blocks[-1].parameters():
                param.requires_grad = True
            for param in self.final_norm.parameters():
                param.requires_grad = True
                
            # unfreeze output head
            for param in self.say_head.parameters():
                param.requires_grad = True 
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.say_head(x)
        return logits
    
    def _load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}

        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
            # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

            # Process the variable name to extract relevant parts
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            # Recursively access or create nested dictionaries
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            # Assign the variable array to the last key
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params
    
    def _load_weights_into_gpt(self, params):
        def assign(left, right):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                                "Right: {right.shape}"
                )
            return torch.nn.Parameter(torch.tensor(right))
        
        # set token and positional embedding weights
        self.pos_emb.weight = assign(self.pos_emb.weight, params['wpe'])
        self.tok_emb.weight = assign(self.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])): # for each transformer block

            # np.split divides weights into three equal parts for query, key, and value components
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            
            # Q, K, V weights
            self.trf_blocks[b].att.W_query.weight = assign(
                self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = assign(
                self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = assign(
                self.trf_blocks[b].att.W_value.weight, v_w.T)
            
            # Q, K, V biases
            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = assign(
                self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = assign(
                self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = assign(
                self.trf_blocks[b].att.W_value.bias, v_b)
            self.trf_blocks[b].att.out_proj.weight = assign(
                self.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.trf_blocks[b].att.out_proj.bias = assign(
                self.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])
            self.trf_blocks[b].ff.layers[0].weight = assign(
                self.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.trf_blocks[b].ff.layers[0].bias = assign(
                self.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.trf_blocks[b].ff.layers[2].weight = assign(
                self.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.trf_blocks[b].ff.layers[2].bias = assign(
                self.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])
            self.trf_blocks[b].norm1.scale = assign(
                self.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            self.trf_blocks[b].norm1.shift = assign(
                self.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            self.trf_blocks[b].norm2.scale = assign(
                self.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            self.trf_blocks[b].norm2.shift = assign(
                self.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])
            
        self.final_norm.scale = assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = assign(self.final_norm.shift, params["b"])

        # weight tying
        # re-use weights of the token embedding layer now in the output layer
        self.out_head.weight = assign(self.out_head.weight, params["wte"])

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = GPT().to(device)