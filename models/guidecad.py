import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from typing import Tuple, List
from models.layers.transformer import *
from models.layers.improved_transformer import *
from models.layers.positional_encoding import *


class ConstEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super(ConstEmbedding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.PE = PositionalEncodingLUT(d_model, max_len=seq_len)

    def forward(self, z):
        batch_size = z.shape[1]
        src = self.PE(z.new_zeros(self.seq_len, batch_size, self.d_model))
        return src


class Decoder(nn.Module):
    def __init__(
        self,
        dim_z: int = 768,
        d_model: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout_f: float = 0.1,
        cad_seq_len: int = 60,
    ):
        super(Decoder, self).__init__()
        self.n_layers_decode = num_layers
        self.d_model = d_model
        self.dim_z = dim_z
        self.n_heads = num_heads
        self.dim_feedforward = ffn_dim
        self.dropout = dropout_f

        self.cad_seq_len = cad_seq_len

        # cad embedding layer
        self.cad_embed = ConstEmbedding(self.d_model, self.cad_seq_len)

        # decoder layer
        decoder_layer = TransformerDecoderLayerGlobalImproved(
            self.d_model, self.dim_z, self.n_heads, self.dim_feedforward, self.dropout
        )
        decoder_norm = LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, self.n_layers_decode, decoder_norm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        src = self.cad_embed(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        out = out.permute(1, 0, 2)
        return out


class AdaptiveLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        input_seq_len: int = 517,
        target_seq_len: int = 256,
        enbed_dim: int = 768,
        proj_dim: int = 768,
    ):
        super(AdaptiveLayer, self).__init__()
        self.target_seq_len = target_seq_len
        if self.target_seq_len > 0:
            # Transformer Encoder (517 → 517)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

            self.adapt_embed = nn.Linear(input_seq_len, target_seq_len)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(enbed_dim, proj_dim)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:

        if self.target_seq_len > 0:
            adapt_embed = self.transformer_encoder(text_embedding)  # [batch_size, 517, 768]
            adapt_embed = adapt_embed.permute(0, 2, 1)  # [batch_size, 768, 517]
            adapt_embed = self.adapt_embed(adapt_embed)  # [batch_size, 768, 256]
            adapt_embed = adapt_embed.permute(0, 2, 1)  # [batch_size, 256, 768]
        else:
            adapt_embed = text_embedding

        adapt_embed = self.pooling(adapt_embed.permute(0, 2, 1))  # [batch_size, 768, 1]
        adapt_embed = adapt_embed.permute(0, 2, 1)  # [batch_size, 1, 768]
        adapt_embed = self.proj(adapt_embed)  # [batch_size, 1, proj_dim]
        return adapt_embed


class TransformerMapper(nn.Module):
    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        num_layers: int = 2,
    ):
        super(TransformerMapper, self).__init__()
        self.prefix_length = prefix_length

        # Use TransformerEncoder instead of full Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=8), num_layers=num_layers
        )
        self.linear = nn.Linear(dim_clip, prefix_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x).view(x.shape[0], self.prefix_length, -1)

        # Combine input and prefix
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        combined = torch.cat((prefix, x), dim=1)  # [batch_size, prefix_length + prefix_length, dim_embedding]

        # Transformer expects [seq_len, batch_size, dim_embedding]
        combined = combined.permute(1, 0, 2)

        # Pass through TransformerEncoder
        out = self.transformer(combined)
        out = out.permute(1, 0, 2)

        # [prefix_length, batch_size, dim_embedding]
        return out[:, self.prefix_length :, :]


class GuideCAD(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        num_seq: int = 60,
        num_commands: int = 6,
        num_params: int = 16,
        param_range: int = 256,
        mapping_num_layers: int = 2,
        prefix_embed_dim: int = 512,
        prefix_length: int = 10,
        d_model: int = 768,
        dim_z: int = 768,
        adapt_embed: int = 256,
        finetune: bool = False,
        token_len: int = 512,
    ):
        super(GuideCAD, self).__init__()
        self.model_name = model_name
        self.prefix_embed_dim = prefix_embed_dim
        self.prefix_length = prefix_length
        # cad command
        self.num_seq = num_seq
        self.num_commands = num_commands
        self.num_params = num_params
        self.param_range = param_range

        # model initialzation
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.gpt_embed = self.gpt.transformer.wte.weight.shape[1]

        if prefix_length > 0:
            # Image embedding projection layer
            self.img_proj = TransformerMapper(
                prefix_embed_dim,
                self.gpt_embed,
                prefix_length,
                mapping_num_layers,
            )

        self.adaptive_layer = AdaptiveLayer(self.gpt_embed, token_len + prefix_length, adapt_embed)

        self.decoder = Decoder(d_model=d_model, dim_z=dim_z)

        # Command prediction
        self.command_head = nn.Linear(self.gpt_embed, num_commands)
        # Parameter prediction
        self.param_head = nn.Linear(self.gpt_embed, num_params * (param_range + 1))

        if not finetune:
            self.freeze_gpt_params()

    def freeze_gpt_params(self):
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.gpt.eval()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [batch_size, seq_len] - Tokenized text input
        attention_mask: [batch_size, seq_len] - Attention mask
        img_embed: [batch_size, img_embed_dim] - Image embeddings
        """

        # --- GPT text embeddings ---
        # text_embeds: [batch_size, seq_len, gpt_hidden_dim]
        text_embeds = self.gpt.transformer.wte(input_ids)
        if self.prefix_length > 0:
            # prefix_token: [batch_size, prefix_length, gpt_hidden_dim]
            prefix_token = self.img_proj(img_embed)
            # combined_embeds: [batch_size, prefix_length + seq_len, gpt_hidden_dim]
            combined_embeds = torch.cat([prefix_token, text_embeds], dim=1)
        else:
            combined_embeds = text_embeds

        gpt_outputs = self.gpt(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        gpt_embed = gpt_outputs.hidden_states[-1]  # 최종 레이어의 hidden state

        # --- Calculate average embedding for text ---
        # [batch_size, seq_len, gpt_hidden_dim]
        pool_embed = self.adaptive_layer(gpt_embed)
        dec_embed = self.decoder(pool_embed.permute(1, 0, 2))

        # command_logits: [batch_size, seq_len, num_commands]
        command_logits = self.command_head(dec_embed)

        # args_logits: [batch_size, seq_len, num_params,  param_range]
        args_logits = self.param_head(dec_embed)
        batch_size = img_embed.shape[0]
        args_logits = args_logits.reshape(batch_size, self.num_seq, self.num_params, -1)

        return command_logits, args_logits, gpt_outputs.logits
