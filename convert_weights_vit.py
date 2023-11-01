#!/usr/bin/env python3
import argparse
import numpy as np
import os
from tfvit import model
from keras.src.utils.data_utils import get_file

BASE_URL = 'https://storage.googleapis.com/vit_models/augreg/{}.npz'
CHECKPOINTS = {
    # https://github.com/google-research/vision_transformer
    'vit_tiny_16_224__imagenet21k': BASE_URL.format('Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0'),
    'vit_tiny_16_384__imagenet': BASE_URL.format('Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0'
                                                 '--imagenet2012-steps_20k-lr_0.03-res_384'),
    'vit_small_16_224__imagenet21k': BASE_URL.format('S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0'),
    'vit_small_16_384__imagenet': BASE_URL.format('S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0'
                                                  '--imagenet2012-steps_20k-lr_0.03-res_384'),
    'vit_small_32_224__imagenet21k': BASE_URL.format('S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0'),
    'vit_small_32_384__imagenet': BASE_URL.format('S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0'
                                                  '--imagenet2012-steps_20k-lr_0.01-res_384'),
    'vit_base_16_224__imagenet21k': BASE_URL.format('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0'),
    'vit_base_16_384__imagenet': BASE_URL.format('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0'
                                                 '--imagenet2012-steps_20k-lr_0.03-res_384'),
    'vit_base_32_224__imagenet21k': BASE_URL.format('B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0'),
    'vit_base_32_384__imagenet': BASE_URL.format('B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0'
                                                 '--imagenet2012-steps_20k-lr_0.01-res_384'),
    'vit_large_16_224__imagenet21k': BASE_URL.format('L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0'),
    'vit_large_16_384__imagenet': BASE_URL.format('L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0'
                                                  '--imagenet2012-steps_20k-lr_0.01-res_384'),
}
MODELS = {
    'vit_tiny_16_224__imagenet21k': model.ViTTiny16224,
    'vit_tiny_16_384__imagenet': model.ViTTiny16384,

    'vit_small_16_224__imagenet21k': model.ViTSmall16224,
    'vit_small_16_384__imagenet': model.ViTSmall16384,

    'vit_small_32_224__imagenet21k': model.ViTSmall32224,
    'vit_small_32_384__imagenet': model.ViTSmall32384,

    'vit_base_16_224__imagenet21k': model.ViTBase16224,
    'vit_base_16_384__imagenet': model.ViTBase16384,

    'vit_base_32_224__imagenet21k': model.ViTBase32224,
    'vit_base_32_384__imagenet': model.ViTBase32384,

    'vit_large_16_224__imagenet21k': model.ViTLarge16224,
    'vit_large_16_384__imagenet': model.ViTLarge16384,
}


def convert_name(n):
    n = f'{n}:0'
    n = n.replace('Transformer/posembed_input/pos_embedding', 'patch/pos/embedding')
    n = n.replace('embedding', 'patch/embed').replace('patch/pos/patch/embed', 'patch/pos/embedding')
    n = n.replace('cls', 'patch/cls/token')
    n = n.replace('Transformer/encoderblock_', 'layer_')
    n = n.replace('LayerNorm_0', 'attn/norm').replace('LayerNorm_2', 'mlp/norm')
    n = n.replace('norm/scale', 'norm/gamma').replace('norm/bias', 'norm/beta')
    n = n.replace('MultiHeadDotProductAttention_1', 'attn/mhsa').replace('out', 'attention_output')
    n = n.replace('MlpBlock_3', 'mlp').replace('Dense_0', 'expand').replace('Dense_1', 'squeeze')
    n = n.replace('head', 'head/proj')
    n = n.replace('Transformer/encoder_norm', 'head/norm')

    return n


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='ViT Transformer weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    model = MODELS[argv.model_type](weights=None)
    path = get_file(fname=None, origin=CHECKPOINTS[argv.model_type], cache_subdir='', cache_dir=argv.out_path)

    weights_jax = np.load(path)
    weights_jax = {convert_name(k): v for k, v in weights_jax.items()}

    weights_tf = []
    for w in model.weights:
        assert w.name in weights_jax, f'Can\'t find weight {w.name} in checkpoint'

        weight = weights_jax[w.name]
        assert w.shape == weight.shape, f'Weight {w.name} shapes not compatible: {w.shape} vs {weight.shape}'

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(path.replace('.npz', '.h5'), save_format='h5')
