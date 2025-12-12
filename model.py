import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Optional, Union
from collections.abc import Callable


# Adapted code from HuggingFace : 
# https://huggingface.co/transformers/v4.9.2/_modules/transformers/models/gpt2/modeling_gpt2.html


class AblatableGPT2Model(GPT2Model):
    def __init__(self, config = None):
        if config is None:
            config = GPT2Config()
        super().__init__(config)

        self.make_ablatable_layers(config)

    def make_ablatable_layers(self, config):
        for layer_idx, block in enumerate(self.h):
            
            attn = block.attn
            block.attn = AblatableGPT2Attention(
                config, 
                is_cross_attention = attn.is_cross_attention,
                layer_idx=layer_idx
            )
            
            mlp = block.mlp
            block.mlp = AblatableGPT2MLP(
                intermediate_size = mlp.c_fc.nf, 
                config = config
            )

    def start_accumulation(self):
        for _, block in enumerate(self.h):
            
            block.attn.reset_accumulation()

            block.mlp.reset_accumulation()



    def update_accumulation_data(self, running_samples_count, abc_means_filepath, abc_means):

        if abc_means is None:

            abc_means = {"last_sample_index" : running_samples_count}
            for i, block in enumerate(self.h):
                
                # Mean of the activations of the two layers of interest
                abc_means[str(i)] = {
                    "attn" : block.attn.accumulation_outputs,
                    "mlp" : block.mlp.accumulation_outputs
                }

                block.attn.reset_accumulation()
                block.mlp.reset_accumulation()

        else:

            abc_means["last_sample_index"] += running_samples_count
            for i, block in enumerate(self.h):
                
                # Mean of the activations of the two layers of interest
                abc_means[str(i)]["attn"] += block.attn.accumulation_outputs
                abc_means[str(i)]["mlp"] += block.mlp.accumulation_outputs

                block.attn.reset_accumulation()
                block.mlp.reset_accumulation()


        torch.save(abc_means, abc_means_filepath)

        print(f"Successfully saved ABC means in {abc_means_filepath}")


        return abc_means



    def ablation(self, all_ablated_means, ablation_config):
        for i, block in enumerate(self.h):
            
            block_ablated_means = all_ablated_means[str(i)]

            heads_ablated_means = block_ablated_means["attention"]

            #mlp_ablated_means = block_ablated_means["mlp"]

            block.attn.set_heads_ablated_means(heads_ablated_means, ablation_config)

            #block.mlp.set_ablated_means(mlp_ablated_means, ablation_config)

    

class AblatableGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.forward_mode = "standard"

    def set_forward_mode(self, forward_mode):
        self.forward_mode = forward_mode

    def set_ablated_means(self, ablated_mean):
        if ablated_mean.dim() == 1:
            ablated_mean = ablated_mean.view(1, 1, -1) 
        self.ablated_mean = ablated_mean

        self.forward_mode = "ablation"

    def reset_accumulation(self):
        self.accumulation_outputs = None

        self.forward_mode = "accumulation"

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:

        if self.forward_mode == "ablation":
            hidden_states = self.ablated_mean.to(hidden_states.dtype).expand_as(hidden_states)
            return hidden_states
        
        hidden_states = super().forward(hidden_states)

        if self.forward_mode == "accumulation":
            if self.accumulation_outputs is None:
                self.accumulation_outputs = hidden_states.detach().sum(dim=0)
            else:
                self.accumulation_outputs += hidden_states.detach().sum(dim=0)

        return hidden_states




class AblatableGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.forward_mode = "standard"


    def set_forward_mode(self, forward_mode):
        self.forward_mode = forward_mode

    def set_heads_ablated_means(self, heads_ablated_means):
        for head_idx, ablated_mean in heads_ablated_means.items():
            if ablated_mean.dim() == 1:
                ablated_mean = ablated_mean.view(1, 1, 1, -1)
                heads_ablated_means[head_idx] = ablated_mean

        self.heads_ablated_means = heads_ablated_means

    def reset_accumulation(self):
        self.accumulation_outputs = None

        self.forward_mode = "accumulation"        


    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )


        
        #####################################################################
        '''
        Replaces the output slice that corresponds to the heads we want to ablate with the
        associated mean tensor from the ABC dataset.
        There is probably a cleaner way to do this, but it is probably good enough for what
        we want to test.
        '''
        if self.forward_mode == "ablation":
            for head_idx, ablated_mean in self.heads_ablated_means:
                attn_output[:, head_idx, :, :] = ablated_mean

        elif self.forward_mode == "accumulation":
            if self.accumulation_outputs is None:
                self.accumulation_outputs = attn_output.detach().sum(dim=0)
            else:
                self.accumulation_outputs += attn_output.detach().sum(dim=0)

        #####################################################################

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights
    







def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights