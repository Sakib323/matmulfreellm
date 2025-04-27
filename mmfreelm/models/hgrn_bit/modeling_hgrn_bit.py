# modeling_hgrn_bit.py

# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import warnings

from typing import List, Optional, Tuple, Union

import torch

import torch.nn as nn

import torch.utils.checkpoint

from transformers.activations import ACT2FN

from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                            CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import GenerationMixin  # Import GenerationMixin

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm
from mmfreelm.modules.activations import swiglu_linear, swiglu

# from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE

logger = logging.get_logger(__name__)


class HGRNBitMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            hidden_ratio: Optional[int] = None,
            intermediate_size: Optional[int] = None,
            hidden_act: str = 'swish'
    ) -> HGRNBitMLP:
        super().__init__()
        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z


class HGRNBitBlock(nn.Module):
    def __init__(self, config: HGRNBitConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx
        )

        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        # Modified MLP initialization
        if config.moe:
            self.mlp = HGRNBitMoE(config)  # Pass the config object
        else:
            self.mlp = HGRNBitMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=config.hidden_ratio,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            lower_bound: Optional[torch.Tensor] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )

        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class HGRNBitPreTrainedModel(PreTrainedModel):
    config_class = HGRNBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
            self,
            module: nn.Module,
            rescale_prenorm_residual: bool = True,
            num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            # > -- GPT-2 :: https://openai.com/blog/better-language-models/

            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class HGRNBitModel(HGRNBitPreTrainedModel):
    def __init__(self, config: HGRNBitConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,  # noqa
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`HGRNBitModel` does not `output_attentions` now, setting it to `False`.")
        output_attentions = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # Attention mask is not used in this model.

        # We can provide a self-attention mask of dimensions [batch_size x key_length x key_length]
        # every item in the diagonal of that matrix should be True.
        # Self-attention will only attend to those position.

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        else:
            expanded_attn_mask = None

        hidden_states = inputs_embeds

        output_shape = input_shape + (hidden_states.size(-1),)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        lower_bound = self.lower_bounds if self.config.use_lower_bound else False

        for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=expanded_attn_mask,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lower_bound=lower_bound
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class HGRNBitForCausalLM(HGRNBitPreTrainedModel, GenerationMixin):  # Inherit from GenerationMixin
    _keys_to_ignore_on_load_missing = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HGRNBitModel(config)
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)
        self.fuse_cross_entropy = config.fuse_cross_entropy

        if self.fuse_cross_entropy:
            self.loss = FusedCrossEntropyLoss(ignore_index=-100)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.fuse_cross_entropy:
                loss = self.loss(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is not causal
        # if attention_mask is not None:
        #     attention_mask = attention_mask[:, : attention_mask.shape[-1]]
        # else:
        #     attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)

        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
