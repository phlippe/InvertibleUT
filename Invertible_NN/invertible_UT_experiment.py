import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

from invertible_UT_model import *

print(problems.available())

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Flags for Tensorflow 1.5 +
FLAGS = tf.flags

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("~/t2t/data/translate_ende_wmt32k")
tmp_dir = os.path.expanduser("~/t2t/tmp")
train_dir = os.path.expanduser("~/t2t/train/UT_invertible")
checkpoint_dir = os.path.expanduser("~/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)

# Fetch the problem
problem_name = "translate_ende_wmt32k"
ende_problem = problems.problem(problem_name)

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_name = "vocab.translate_ende_wmt32k.32768.subwords"
vocab_file = os.path.join(data_dir, vocab_name)

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(data_dir)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))

# There are many models available in Tensor2Tensor
print(registry.list_models())

# Create hparams and the model
model_name = "invertible_ut"
hparams_set = "universal_transformer_tiny"

hparams = trainer_lib.create_hparams(hparams_set)


FLAGS.problems = problem_name
FLAGS.model = model_name
FLAGS.schedule = "train_and_evaluate"
FLAGS.save_checkpoints_secs = 0
FLAGS.local_eval_frequency = 2000
FLAGS.gpu_memory_fraction = .99
FLAGS.worker_gpu = 1
FLAGS.ps_gpu = 2
FLAGS.log_device_placement = True
FLAGS.worker_replicas = 2

RUN_CONFIG = trainer_lib.create_run_config(
      model_dir=train_dir,
      model_name="test",
      keep_checkpoint_max=3,
      save_checkpoints_secs=0,
      gpu_mem_fraction=FLAGS.gpu_memory_fraction
)


exp_fn = trainer_lib.create_experiment(
        run_config=RUN_CONFIG,
        hparams=hparams,
        model_name=model_name,
        problem_name=problem_name,
        data_dir=(data_dir),
        train_steps=1000000,
        eval_steps=100
    )
exp_fn.train_and_evaluate()