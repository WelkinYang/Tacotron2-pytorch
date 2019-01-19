from tf_hparams import HParams

# Copyright 2018 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)

# Default hyperparameters:
hparams = HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',

    lang="enus",

    # Audio:
    num_mels=80,
    num_freq=1025,
    min_mel_freq=0,
    max_mel_freq=8000,
    sample_rate=16000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasize=0.97,
    min_level_db=-100,
    ref_level_db=20,  # suggest use 20 for griffin-lim and 0 for wavenet
    max_abs_value=1,
    symmetric_specs=False,  # if true, suggest use 4 as max_abs_value

    # Model:
    use_phone=False,

    outputs_per_step=2,
    embedding_dim=512,

    encoder_conv_layers=3,
    encoder_conv_width=5,
    encoder_conv_channels=512,
    encoder_lstm_units=256,  # For each direction

    attention_depth=128,
    attention_filters = 32,
    attention_kernel = 31,
    attention_type="soft",
    cumulate_weights=True,  # Whether to cumulate all previous attention weights

    decoder_lstm_layers=2,
    decoder_lstm_units=1024,
    zoneout_prob_cells=0.1,
    zoneout_prob_states=0.1,

    postnet_conv_layers=5,
    postnet_conv_width=5,
    postnet_conv_channels=512,

    dropout_rate=0.5,

    prenet_inference_dropout=True,

    # Training:
    gpu_ids='0,1',
    batch_size=16,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    warmup_steps=50000,
    initial_learning_rate=0.001,
    final_learning_rate=1e-5,
    decay_rate=0.5,  # learning rate decay rate
    clip_norm=1.0,  # gradient clip to avoid explosion
    shuffle_training_data=True,  # make training data shuffled or sorted
    keep_training_set_ratio=1.,   # remove too long sequence in training set to avoid OOM,
                                  # depends on your training set/receipe, GPU

    use_l2_regularization=False,
    l2_weight=1e-6,
    use_vector_quantization=False,
    vq_dim=512,
    use_linear_spec=True,  # Predict mel spec or linear spec
    use_stop_token=True,
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    use_gta_mode = True, #Use ground_truth_align
    teacher_forcing_ratio=1.,
    teacher_forcing_schema="full",

    # Test:
    test_batch_size=5,

    # Eval:
    max_iters=3000,  # max decode iterations, 100 means max wave length ceiling is 1.25s if frame_shift_ms=12.5
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim

    stop_at_any=True,
    predict_extra_frame=5,
    fixed_random_seed=False,
    random_seed=20180705,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)