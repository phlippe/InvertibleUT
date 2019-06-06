import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections
import copy
import functools
import json
import sys

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.models.research import universal_transformer_util, universal_transformer
from tensor2tensor.models.research.universal_transformer_util import step_preprocess
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils


@registry.register_model
class InvertibleUT(universal_transformer.UniversalTransformer):
  """Universal Transformer: Depth-wise recurrent transformer model."""

  def encode(self, inputs, target_space, hparams, features=None, losses=None,
             **kwargs):
    """Encode Universal Transformer inputs.
    It is similar to "transformer.encode", but it uses
    "universal_transformer_util.universal_transformer_encoder" instead of
    "transformer.transformer_encoder".
    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.
      **kwargs: additional arguments to pass to encoder_function
    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    
    ####
    ## DEBUG
    ####
    # with open("invertible_UT_params.json", "w") as f:
    #   json.dump(dict(hparams.__dict__), f, default=lambda o: '<not serializable>', sort_keys=True,
    #             indent=4, separators=(',', ': '))
    # sys.exit()

    del losses

    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output, encoder_extra_output) = (
        invertible_UT_encoder(
            encoder_input,
            self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights))

    return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             ** kwargs):
    """Decode Universal Transformer outputs from encoder representation.
    It is similar to "transformer.decode", but it uses
    "universal_transformer_util.universal_transformer_decoder" instead of
    "transformer.transformer_decoder".
    Args:
      decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
        hidden_dim]
      encoder_output: Encoder representation. [batch_size, input_length,
        hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
        attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: Unimplemented.
      decode_loop_step: Unused.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: Unused.
      **kwargs: additional arguments to pass to decoder_function
    Returns:
       Tuple of:
         Final decoder representation. [batch_size, decoder_length,
            hidden_dim]
         encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    del decode_loop_step
    del losses
    # TODO(dehghani): enable caching.
    del cache

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    # No caching in Universal Transformers!
    (decoder_output, dec_extra_output) = (
        invertible_UT_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            nonpadding=nonpadding,
            save_weights_to=self.attention_weights))

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2), dec_extra_output


###################################################
## Functions copied from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py
## and slightly adapted for invertible UT. 
## TODO: Change comments!
###################################################



def invertible_UT_encoder(encoder_input,
                          encoder_self_attention_bias,
                          hparams,
                          name="encoder",
                          nonpadding=None,
                          save_weights_to=None,
                          make_image_summary=True):
  """Invertible Universal Transformer encoder function.
  Prepares all the arguments and the inputs and passes it to a
  universal_transformer_layer to encode the encoder_input.
  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
  Returns:
    y: a Tensors as the output of the encoder
    extra_output: which can be used to pass extra information to the body
  """

  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)

    ##################
    ## CHANGE START ##
    ##################

    # From now on, ffn_unit and attention_unit are lists with the number of layers/splits we actually have
    ffn_unit = []
    attention_unit = []

    sub_hparams = copy.deepcopy(hparams)
    sub_hparams.hidden_size /= 2

    for i in range(2):
      with tf.variable_scope("part_%i" % i):

        ffn_unit.append(functools.partial(
            invertible_transformer_encoder_ffn_unit,
            hparams=sub_hparams,
            nonpadding_mask=nonpadding,
            pad_remover=pad_remover,
            split_index=i))

        attention_unit.append(functools.partial(
            invertible_transformer_encoder_attention_unit,
            hparams=sub_hparams,
            encoder_self_attention_bias=encoder_self_attention_bias,
            attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            split_index=i))

    x, extra_output = invertible_UT_layer(
        x, hparams, ffn_unit, attention_unit, pad_remover=pad_remover)

    ##################
    ##  CHANGE END  ##
    ##################

    return common_layers.layer_preprocess(x, hparams), extra_output


def invertible_UT_decoder(decoder_input,
                                  encoder_output,
                                  decoder_self_attention_bias,
                                  encoder_decoder_attention_bias,
                                  hparams,
                                  name="decoder",
                                  nonpadding=None,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """Universal Transformer decoder function.
  Prepares all the arguments and the inputs and passes it to a
  core_universal_transformer_layer to decoder.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convoltutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
  Returns:
    y: the output Tensors
    extra_output: which can be used to pass extra information to the body
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):

    ##################
    ## CHANGE START ##
    ##################

    # From now on, ffn_unit and attention_unit are lists with the number of layers/splits we actually have
    ffn_unit = []
    attention_unit = []

    sub_hparams = copy.deepcopy(hparams)
    sub_hparams.hidden_size /= 2

    for i in range(2):
      with tf.variable_scope("part_%i" % i):
        ffn_unit.append(functools.partial(
            invertible_transformer_decoder_ffn_unit,
            hparams=sub_hparams,
            nonpadding_mask=nonpadding,
            split_index=i))

        attention_unit.append(functools.partial(
            invertible_transformer_decoder_attention_unit,
            hparams=sub_hparams,
            encoder_output=encoder_output,
            decoder_self_attention_bias=decoder_self_attention_bias,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            split_index=i))

    x, extra_output = invertible_UT_layer(
        x, hparams, ffn_unit, attention_unit)

    ##################
    ##  CHANGE END  ##
    ##################

    return common_layers.layer_preprocess(x, hparams), extra_output


def invertible_UT_layer(x,
                        hparams,
                        ffn_unit,
                        attention_unit,
                        pad_remover=None):
  """Core function applying the universal transformer layer.
  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).
  Returns:
    the output tensor,  extra output (can be memory, ponder time, etc.)
  Raises:
    ValueError: Unknown recurrence type
  """

  ############
  ## No changes as this would be a transformer applied before the actual UT
  ############
  def add_vanilla_transformer_layer(x, num_layers, name):
    """Passes the input through num_layers of vanilla transformer layers.
    Args:
     x: input
     num_layers: number of layers
     name: string, prefix of layer names
    Returns:
       output of vanilla_transformer_layer
    """
    if hparams.add_position_timing_signal:
      # In case of add_position_timing_signal=true, we set  hparams.pos=None
      # and add position timing signal at the beginning of each step, so for
      # the vanilla transformer, we need to add timing signal here.
      x = common_attention.add_timing_signal_1d(x)
    for layer in range(num_layers):
      with tf.variable_scope(name + "layer_%d" % layer):
        x = ffn_unit(attention_unit(x))
    return x

  with tf.variable_scope("universal_transformer_%s" % hparams.recurrence_type):
    if (hparams.mix_with_transformer and
        "before_ut" in hparams.mix_with_transformer):
      x = add_vanilla_transformer_layer(x, hparams.num_mixedin_layers,
                                        "before_ut_")

    if hparams.recurrence_type == "act":
      # TODO: Implement this function as well!
      output, extra_output = universal_transformer_act(
          x, hparams, ffn_unit, attention_unit)

    else:  # for all the other recurrency types with fixed number of steps

      ##################
      ## CHANGE START ##
      ##################

      ut_function, initializer = get_invertible_ut_layer(x, hparams, ffn_unit,
                                                     attention_unit, pad_remover)

      ##################
      ##  CHANGE END  ##
      ##################

      output, _, extra_output = tf.foldl(
          ut_function, tf.range(hparams.num_rec_steps),
          initializer=initializer)

      # Right now, this is only possible when the transition function is an lstm
      if (hparams.recurrence_type == "lstm" and
          hparams.get("use_memory_as_final_state", False)):
        output = extra_output

    if (hparams.mix_with_transformer and
        "after_ut" in hparams.mix_with_transformer):
      output = add_vanilla_transformer_layer(output, hparams.num_mixedin_layers,
                                             "after_ut_")

    return output, extra_output


def get_invertible_ut_layer(x,
                      hparams,
                      ffn_unit,
                      attention_unit,
                      pad_remover=None):
  """Provides the function that is used in universal transforemr steps.
  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).
  Returns:
    ut_function and the ut_initializer
  Raises:
    ValueError: Unknown recurrence type
  """

  if hparams.recurrence_type == "basic":

    ##################
    ## CHANGE START ##
    ##################
    
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        invertible_universal_transformer_basic,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

    ##################
    ##  CHANGE END  ##
    ##################

  elif hparams.recurrence_type == "highway":
    # TODO: Implement this variant of UT!
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_highway,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "skip":
    # TODO: Implement this variant of UT!
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_skip,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "dwa":
    # TODO: Implement this variant of UT!

    # memory contains the original input + all the states
    memory_size = hparams.num_rec_steps + 1

    # prepare initializer:
    memory_empty = tf.zeros([memory_size] + common_layers.shape_list(x))

    # filling the first slot with the original input
    memory = fill_memory_slot(memory_empty, x, 0)

    ut_initializer = (x, x, memory)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_depthwise_attention,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  elif hparams.recurrence_type == "gru":
    # TODO: Implement this variant of UT!

    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_gru_as_transition_function,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "lstm":
    # TODO: Implement this variant of UT!

    memory = tf.zeros(common_layers.shape_list(x))
    ut_initializer = (x, x, memory)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_lstm_as_transition_function,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  else:
    raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)

  return ut_function, ut_initializer


def invertible_transformer_encoder_ffn_unit(x,
                                            hparams,
                                            nonpadding_mask=None,
                                            pad_remover=None,
                                            split_index=0):
  """Applies a feed-forward function which is parametrised for encoding.
  Args:
    x: input
    hparams: model hyper-parameters
    nonpadding_mask: optional Tensor with shape [batch_size, encoder_length]
    indicating what positions are not padding.  This is used
    to mask out padding in convoltutional layers.  We generally only
    need this mask for "packed" datasets, because for ordinary datasets,
    no padding is ever followed by nonpadding.
    pad_remover: to mask out padding in convolutional layers (efficiency).
  Returns:
    the output tensor
  """

  with tf.variable_scope("ffn"):

    ##################
    ## CHANGE START ##
    ##################

    x_splits = tf.split(x, num_or_size_splits=2, axis=2)

    if hparams.transformer_ffn_type == "fc":
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x_splits[split_index], hparams),
          hparams,
          pad_remover,
          conv_padding="SAME",
          nonpadding_mask=nonpadding_mask)

    if hparams.transformer_ffn_type == "sepconv":
      assert nonpadding_mask is not None, (
          "The nonpadding_mask should be provided, otherwise the model uses "
          "the leaked padding information to estimate the length!")
      y = common_layers.sepconv_relu_sepconv(
          common_layers.layer_preprocess(x_splits[split_index], hparams),
          filter_size=hparams.filter_size,
          output_size=hparams.hidden_size,
          first_kernel_size=(3, 1),
          second_kernel_size=(5, 1),
          padding="SAME",
          nonpadding_mask=nonpadding_mask,
          dropout=hparams.relu_dropout)

    x_splits[1 - split_index] = common_layers.layer_postprocess(x_splits[1 - split_index], y, hparams)
    x = tf.concat(x_splits, axis=2)

    ##################
    ##  CHANGE END  ##
    ##################

  return x


def invertible_transformer_encoder_attention_unit(x,
                                                 hparams,
                                                 encoder_self_attention_bias,
                                                 attention_dropout_broadcast_dims,
                                                 save_weights_to=None,
                                                 make_image_summary=True,
                                                 split_index=0):
  """Applies multihead attention function which is parametrised for encoding.
  Args:
    x: input
    hparams: model hyper-parameters
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    attention_dropout_broadcast_dims: Fpr noise broadcasting in the dropout
      layers to save memory during training
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
  Returns:
    the output tensor
  """

  with tf.variable_scope("self_attention"):

    ##################
    ## CHANGE START ##
    ##################

    x_splits = tf.split(x, num_or_size_splits=2, axis=2)

    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x_splits[split_index], hparams),
        None,
        encoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        save_weights_to=save_weights_to,
        max_relative_position=hparams.max_relative_position,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=attention_dropout_broadcast_dims,
        hard_attention_k=hparams.hard_attention_k)

    x_splits[1 - split_index] = common_layers.layer_postprocess(x_splits[1 - split_index], y, hparams)
    x = tf.concat(x_splits, axis=2)

    ##################
    ##  CHANGE END  ##
    ##################

  return x


def invertible_transformer_decoder_ffn_unit(x,
                                           hparams,
                                           nonpadding_mask=None,
                                           split_index=0):
  """Applies a feed-forward function which is parametrised for decoding.
  Args:
    x: input
    hparams: model hyper-parameters
    nonpadding_mask: optional Tensor with shape [batch_size, encoder_length]
    indicating what positions are not padding.  This is used
    to mask out padding in convoltutional layers.  We generally only
    need this mask for "packed" datasets, because for ordinary datasets,
    no padding is ever followed by nonpadding.
  Returns:
    the output tensor
  """

  with tf.variable_scope("ffn"):

    ##################
    ## CHANGE START ##
    ##################

    x_splits = tf.split(x, num_or_size_splits=2, axis=2)

    if hparams.transformer_ffn_type == "fc":
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x_splits[split_index], hparams),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding_mask)

    if hparams.transformer_ffn_type == "sepconv":
      y = common_layers.sepconv_relu_sepconv(
          common_layers.layer_preprocess(x_splits[split_index], hparams),
          filter_size=hparams.filter_size,
          output_size=hparams.hidden_size,
          first_kernel_size=(3, 1),
          second_kernel_size=(5, 1),
          padding="LEFT",
          nonpadding_mask=nonpadding_mask,
          dropout=hparams.relu_dropout)

    x_splits[1 - split_index] = common_layers.layer_postprocess(x_splits[1 - split_index], y, hparams)
    x = tf.concat(x_splits, axis=2)

    ##################
    ##  CHANGE END  ##
    ##################

  return x


def invertible_transformer_decoder_attention_unit(x,
                                                 hparams,
                                                 encoder_output,
                                                 decoder_self_attention_bias,
                                                 encoder_decoder_attention_bias,
                                                 attention_dropout_broadcast_dims,
                                                 save_weights_to=None,
                                                 make_image_summary=True,
                                                 split_index=0):
  """Applies multihead attention function which is parametrised for decoding.
  Args:
    x: input (decoder input)
    hparams: model hyper-parameters
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    attention_dropout_broadcast_dims: Fpr noise broadcasting in the dropout
      layers to save memory during training
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
  Returns:
    The output tensor
  """

  ##################
  ## CHANGE START ##
  ##################

  with tf.variable_scope("self_attention"):

    # ERROR: Output is 128 instead of 64

    x_splits = tf.split(x, num_or_size_splits=2, axis=2)

    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x_splits[split_index], hparams),
        None,
        decoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        save_weights_to=save_weights_to,
        max_relative_position=hparams.max_relative_position,
        cache=None,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=attention_dropout_broadcast_dims,
        hard_attention_k=hparams.hard_attention_k)

    x_splits[1 - split_index] = common_layers.layer_postprocess(x_splits[1 - split_index], y, hparams)
    
  if encoder_output is not None:
    with tf.variable_scope("encdec_attention"):
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(x_splits[split_index], hparams),
          encoder_output,
          encoder_decoder_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size,
          hparams.num_heads,
          hparams.attention_dropout,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=attention_dropout_broadcast_dims,
          hard_attention_k=hparams.hard_attention_k)
      
      x_splits[1 - split_index] = common_layers.layer_postprocess(x_splits[1 - split_index], y, hparams)
    
  x = tf.concat(x_splits, axis=2)

  ##################
  ##  CHANGE END  ##
  ##################

  return x


def invertible_universal_transformer_basic(layer_inputs,
                                           step, hparams,
                                           ffn_unit,
                                           attention_unit):
  """Basic Universal Transformer.
  This model is pretty similar to the vanilla transformer in which weights are
  shared between layers. For some tasks, this simple idea brings a
  generalization that is not achievable by playing with the size of the model
  or drop_out parameters in the vanilla transformer.
  Args:
    layer_inputs:
        - state: state
    step: indicates number of steps taken so far
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
  Returns:
    layer_output:
         new_state: new state
  """
  state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0,
                                     name="unstack")
  new_state = step_preprocess(state, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      if isinstance(ffn_unit, list) and isinstance(attention_unit, list):
        for sub_ffn_unit, sub_attention_unit in zip(ffn_unit, attention_unit):
          new_state = sub_ffn_unit(sub_attention_unit(new_state))
      else:
        new_state = ffn_unit(attention_unit(new_state))

  return new_state, inputs, memory