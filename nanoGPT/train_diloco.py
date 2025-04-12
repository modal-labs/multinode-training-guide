"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset
import os
import wandb


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    @classmethod
    def gpt2_small(cls):
        return cls(n_layer=4, n_head=8, n_embd=256)

    @classmethod
    def gpt2_base(cls):
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def gpt2_medium(cls):
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def gpt2_large(cls):
        return cls(n_layer=36, n_head=20, n_embd=1280)

    @classmethod
    def gpt2_xl(cls):
        return cls(n_layer=48, n_head=25, n_embd=1600)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, inference=False):
        device = idx.device
        assert idx.ndim == 2, f"Expected input of shape [B, T], got shape {idx.shape}"
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if inference:
            x = x[:, [-1], :]
            # if we are given some desired targets also calculate the loss
        logits = self.lm_head(x)
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, inference=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# add diloco support
def identity_loss(x, _):
    return x


class DilocoGPTWrapper(GPT):
    """Minimal wrapper around GPT to make it compatible with diloco while preserving all original functionality"""

    def forward(self, batch):
        # Unpack the batch tuple - batch[0] contains the input tensor
        idx = batch[0].to(self.lm_head.weight.device)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        # Get logits using parent's forward
        logits = super().forward(idx)

        # Calculate loss using original nanoGPT logic
        b, t, c = logits.shape
        logits = logits.view(b * t, c)
        targets = torch.roll(idx, shifts=-1, dims=1)
        targets[:, -1] = -1  # The last prediction is not valid
        targets = targets.contiguous().view(b * t)
        loss = nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        return loss  # Return scalar loss directly


class NanoGPTTrainer(DilocoSimulator):
    """DilocoSimulator with WandB logging for nanoGPT"""

    def _eval_model(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                batch = self._get_batch(eval=True)
                loss = self.model(batch)
                total_loss += loss.item()

                if self.rank == 0:
                    wandb.log(
                        {
                            "eval_loss": loss.item(),
                            "eval_perplexity": torch.exp(
                                torch.tensor(loss.item())
                            ).item(),
                        }
                    )

        avg_loss = total_loss / self.config.eval_iters
        if self.rank == 0:
            print(f"Eval Loss: {avg_loss}")

        self.model.train()
        return avg_loss

    def _outer_step(self) -> None:
        if self.rank == 0:
            print(f"iter {self.local_step}: running outer step")
        super()._outer_step()

    def _train_step(self):
        batch = self._get_batch()
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.rank == 0:
            self.training_time_ms += 1000 * (
                time.perf_counter() - self.training_time_t0
            )
            train_loss = loss.item()
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_perplexity": torch.exp(torch.tensor(train_loss)).item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "step": self.local_step,
                }
            )
            print(
                f"iter {self.local_step}: loss {train_loss:.4f} "
                f"train_time:{self.training_time_ms:.0f}ms step_avg:{self.training_time_ms/max(self.local_step, 1):.2f}ms"
            )
            self.training_time_t0 = time.perf_counter()

    def _train(self, rank: int):
        self.training_time_ms = 0
        self.training_time_t0 = time.perf_counter()
        if rank == 0:
            wandb.init(project="nanogpt-diloco")
        try:
            super()._train(rank)
        finally:
            if rank == 0:
                wandb.finish()


class GPTTrainDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for training data"""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        return x, x


def get_dataset(args):
    print(f"Loading dataset: {args.dataset}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.dataset == "shakespeare":
        dataset = load_dataset("Trelis/tiny-shakespeare")
        text_column = "Text"
    elif args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text_column = "text"
    elif args.dataset == "code":
        dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train[:1%]")
        text_column = "content"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset loaded. Structure: {dataset}")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column], truncation=True, max_length=args.block_size
        )

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["train"].column_names
    )

    # Convert to torch tensors
    def convert_to_tensor(examples):
        all_ids = []
        for ids in examples["input_ids"]:
            if len(ids) > 0:  # Skip empty sequences
                all_ids.extend(ids + [tokenizer.eos_token_id])

        if len(all_ids) == 0:
            return {"input_ids": torch.tensor([])}

        # Make sure we have complete blocks
        num_blocks = len(all_ids) // args.block_size
        if num_blocks == 0:
            return {"input_ids": torch.tensor([])}

        all_ids = all_ids[: num_blocks * args.block_size]
        tensor_data = torch.tensor(all_ids).view(-1, args.block_size)
        return {"input_ids": tensor_data}

    print("Converting to tensors...")
    tensor_dataset = tokenized_dataset.map(
        convert_to_tensor,
        batched=True,
        remove_columns=tokenized_dataset["train"].column_names,
    )

    # Extract and concatenate all tensors
    train_data = torch.cat(
        [
            torch.as_tensor(x)
            for x in tensor_dataset["train"]["input_ids"]
            if len(x) > 0
        ],
        dim=0,
    )
    val_data = torch.cat(
        [torch.as_tensor(x) for x in tensor_dataset["test"]["input_ids"] if len(x) > 0],
        dim=0,
    )

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")
    return train_data, val_data, tokenizer.vocab_size


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="which dataset to use (shakespeare, wikitext, code)",
    )
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load dataset from HuggingFace
    train_data, val_data, vocab_size = get_dataset(args)

    # Create datasets
    train_dataset = GPTTrainDataset(train_data, args.block_size)
    val_dataset = GPTTrainDataset(val_data, args.block_size)

    # Create model config
    gpt_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=12,  # Use smaller model for testing, adjust as needed
        n_head=12,
        n_embd=768,
    )

    # Create diloco config
    diloco_config = DilocoSimulatorConfig(
        model_cls=DilocoGPTWrapper,
        model_kwargs={"config": gpt_config},
        optimizer_kwargs={
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
        },
        loss_fn=identity_loss,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        eval_iters=200,
        ckpt_interval=1000,
        num_nodes=args.num_nodes,
        diloco_interval=1000,
        devices=list(range(torch.cuda.device_count())),
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize trainer and start training
    print("Initializing trainer...")
    trainer = NanoGPTTrainer(diloco_config)
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
