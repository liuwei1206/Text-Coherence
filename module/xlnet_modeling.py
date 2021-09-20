# author = liuwei
# date = 2021-09-17

"""copy from the implementation of bert in transformers"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_utils import (
    PoolerAnswerClass,
    PoolerEndLogits,
    PoolerStartLogits,
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
)
from transformers.utils import logging
from transformers.models.xlnet.configuration_xlnet import XLNetConfig

def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch. I use a map to keep the PyTorch model as identical to the original PyTorch
    model as possible.
    """

    tf_to_pt_map = {}

    if hasattr(model, "transformer"):
        if hasattr(model, "lm_loss"):
            # We will load also the output bias
            tf_to_pt_map["model/lm_loss/bias"] = model.lm_loss.bias
        if hasattr(model, "sequence_summary") and "model/sequnece_summary/summary/kernel" in tf_weights:
            # We will load also the sequence summary
            tf_to_pt_map["model/sequnece_summary/summary/kernel"] = model.sequence_summary.summary.weight
            tf_to_pt_map["model/sequnece_summary/summary/bias"] = model.sequence_summary.summary.bias
        if (
            hasattr(model, "logits_proj")
            and config.finetuning_task is not None
            and f"model/regression_{config.finetuning_task}/logit/kernel" in tf_weights
        ):
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/kernel"] = model.logits_proj.weight
            tf_to_pt_map[f"model/regression_{config.finetuning_task}/logit/bias"] = model.logits_proj.bias

        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings and output
    tf_to_pt_map.update(
        {
            "model/transformer/word_embedding/lookup_table": model.word_embedding.weight,
            "model/transformer/mask_emb/mask_emb": model.mask_emb,
        }
    )

    # Transformer blocks
    for i, b in enumerate(model.layer):
        layer_str = f"model/transformer/layer_{i}/"
        tf_to_pt_map.update(
            {
                layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
                layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
                layer_str + "rel_attn/o/kernel": b.rel_attn.o,
                layer_str + "rel_attn/q/kernel": b.rel_attn.q,
                layer_str + "rel_attn/k/kernel": b.rel_attn.k,
                layer_str + "rel_attn/r/kernel": b.rel_attn.r,
                layer_str + "rel_attn/v/kernel": b.rel_attn.v,
                layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
                layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
                layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
                layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
                layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
                layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
            }
        )

    # Relative positioning biases
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update(
        {
            "model/transformer/r_r_bias": r_r_list,
            "model/transformer/r_w_bias": r_w_list,
            "model/transformer/r_s_bias": r_s_list,
            "model/transformer/seg_embed": seg_embed_list,
        }
    )
    return tf_to_pt_map


def load_tf_weights_in_xlnet(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if "kernel" in name and ("ff" in name or "summary" in name or "logit" in name):
            logger.info("Transposing")
            array = np.transpose(array)
        if isinstance(pointer, list):
            # Here we will split the TF weights
            assert (
                len(pointer) == array.shape[0]
            ), f"Pointer length {len(pointer)} and array length {array.shape[0]} mismatched"
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert (
                        p_i.shape == arr_i.shape
                    ), f"Pointer shape {p_i.shape} and array shape {arr_i.shape} mismatched"
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f"Initialize PyTorch weight {name} for layer {i}")
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert (
                    pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {name}")
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/Adam", None)
        tf_weights.pop(name + "/Adam_1", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=None,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum("ijbs,ibns->bnij", seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        # attention probability
        attn_prob = nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum("ijbn->bnij", head_mask)

        # attention output
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, torch.einsum("bnij->ijbn", attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)

            # content-based value head
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)

            # position-based key head
            k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)

            # h-stream
            # content-stream query head
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            # g-stream
            # query-stream query head
            q_head_g = torch.einsum("ibh,hnd->ibnd", g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            # Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content heads
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)

            # positional heads
            # type casting for fp16 support
            k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        outputs = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = apply_chunking_to_forward(
                self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g
            )
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs

    def ff_chunk(self, output_x):
        output_x = self.ff(output_x)
        return output_x


class XLNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XLNetConfig
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.o,
                module.r,
                module.r_r_bias,
                module.r_s_bias,
                module.r_w_bias,
                module.seg_embed,
            ]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel):
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)


@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetModel`.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config):
        """
        This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
        methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
        pruning heads etc.)

        This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
        subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
        general usage and behavior.

        Parameters:
            config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
                weights.
        """
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: Sequence length
            mlen: Mask length

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(self.device)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        # cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if self.mem_len is None or self.mem_len == 0:
            # If :obj:`use_mems` is active but no `mem_len` is defined, the model behaves like GPT-2 at inference time
            # and returns all of the past and current hidden states.
            cutoff = 0
        else:
            # If :obj:`use_mems` is active and `mem_len` is defined, the model returns the last `mem_len` hidden
            # states. This is the preferred setting for training and long-form generation.
            cutoff = -self.mem_len
        if prev_mem is None:
            # if :obj:`use_mems` is active and `mem_len` is defined, the model
            new_mem = curr_out[cutoff:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(self.device)
        return pos_emb

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  # delete after depreciation warning is removed
    ):
        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using :class:`transformers.XLNetTokenizer`. See
                :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
                details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
                Contains pre-computed hidden-states (see :obj:`mems` output below) . Can be used to speed up sequential
                decoding. The token ids which have their past given to this model should not be passed as :obj:`input_ids`
                as they have already been computed.

                :obj:`use_mems` has to be set to :obj:`True` to make use of :obj:`mems`.
            perm_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, sequence_length)`, `optional`):
                Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:

                - if ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
                - if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.

                If not set, each token attends to all the others (full bidirectional attention). Only used during
                pretraining (to define factorization order) or for sequential decoding (generation).
                仅仅在预训练阶段使用
            target_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, sequence_length)`, `optional`):
                Mask to indicate the output tokens to use. If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k
                is on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
                (generation).
                仅仅在预训练阶段使用
            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
                1]``:

                - 0 corresponds to a `sentence A` token,
                - 1 corresponds to a `sentence B` token.

                `What are token type IDs? <../glossary.html#token-type-ids>`__
            input_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
                Mask to avoid performing attention on padding token indices. Negative of :obj:`attention_mask`, i.e. with 0
                for real tokens and 1 for padding which is kept for compatibility with the original code base.

                Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **masked**,
                - 0 for tokens that are **not masked**.

                You can only uses one of :obj:`input_mask` and :obj:`attention_mask`.

                与attention_mask功能一样，只是与其刚好相反，即这里0表示有效的词，1表示无效的词，这个主要是
                为了和原始代码统一
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
                vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
                tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
                more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

        Examples::

            >>> from transformers import XLNetTokenizer, XLNetLMHeadModel
            >>> import torch

            >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            >>> model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            >>> next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

            >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
            >>> assert labels.shape[0] == 1, 'only one word will be predicted'
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
            >>> loss = outputs.loss
            >>> next_token_logits = outputs.logits  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems` instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = nn.functional.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()

        if not use_mems:
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )

