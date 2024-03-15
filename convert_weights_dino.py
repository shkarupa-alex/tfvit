#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
from tfvit import model
from tf_keras.src.utils.data_utils import get_file

BASE_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_{}.pth'
CHECKPOINTS = {
    # https://github.com/facebookresearch/dinov2
    'vit_small_14_518__dino2reg': [
        BASE_URL.format('vits14/dinov2_vits14_reg4_pretrain'), BASE_URL.format('vits14/dinov2_vits14_reg4_linear_head')
    ],
    'vit_base_14_518__dino2reg': [
        BASE_URL.format('vitb14/dinov2_vitb14_reg4_pretrain'), BASE_URL.format('vitb14/dinov2_vitb14_reg4_linear_head')
    ],
    'vit_large_14_518__dino2reg': [
        BASE_URL.format('vitl14/dinov2_vitl14_reg4_pretrain'), BASE_URL.format('vitl14/dinov2_vitl14_reg4_linear_head')
    ],
    'vit_giant_14_518__dino2reg': [
        BASE_URL.format('vitg14/dinov2_vitg14_reg4_pretrain'), BASE_URL.format('vitg14/dinov2_vitg14_reg4_linear_head')
    ],
}
MODELS = {
    'vit_small_14_518__dino2reg': model.ViTSmall14518,
    'vit_base_14_518__dino2reg': model.ViTBase14518,
    'vit_large_14_518__dino2reg': model.ViTLarge14518,
    'vit_giant_14_518__dino2reg': model.ViTGiant14518
}


def transform_weights(weights, embed, heads):
    head_dim = embed // heads

    weights.pop('mask_token:0')

    weights['patch/cls/token:0'] = np.concatenate([
        weights['patch/cls/token:0'], weights.pop('register_tokens:0')], axis=1)

    weights['patch/pos/embedding:0'] = np.concatenate([
        weights['patch/pos/embedding:0'][:, :1], np.zeros([1, 4, embed], 'float32'),
        weights['patch/pos/embedding:0'][:, 1:]], axis=1)

    for key in list(weights.keys()):
        value = weights[key]

        if 'patch/embed/kernel' in key:
            value = value.transpose(2, 3, 1, 0)

        if '/scale/' in key:
            value = value[None, None]

        if any([part in key for part in {'/value/', '/attention_output/', '/expand/', '/squeeze/', '/proj/'}]):
            value = value.T

        if '/value/bias:0' in key:
            value = value.reshape(3, heads, head_dim)
            weights[key.replace('/value/', '/query/')] = value[0]
            weights[key.replace('/value/', '/key/')] = value[1]
            value = value[2]

        if '/value/kernel:0' in key:
            value = value.reshape(embed, 3, heads, head_dim)
            weights[key.replace('/value/', '/query/')] = value[:, 0]
            weights[key.replace('/value/', '/key/')] = value[:, 1]
            value = value[:, 2]

        if '/attention_output/kernel:0' in key:
            value = value.reshape(heads, head_dim, embed)

        weights[key] = value

    return weights


def convert_name(n):
    n = f'{n}:0'
    n = n.replace('blocks.', 'layer_')
    n = n.replace('.norm1.weight', '/attn/norm/gamma').replace('.norm1.bias', '/attn/norm/beta')
    n = n.replace('.norm2.weight', '/mlp/norm/gamma').replace('.norm2.bias', '/mlp/norm/beta')
    n = n.replace('.attn.qkv.', '/attn/mhsa/value/').replace('.attn.proj.', '/attn/mhsa/attention_output/')
    n = n.replace('.mlp.fc1.', '/mlp/expand/').replace('.mlp.fc2.', '/mlp/squeeze/')
    n = n.replace('.mlp.w12.', '/mlp/expand/').replace('.mlp.w3.', '/mlp/squeeze/')
    n = n.replace('.ls1.', '/attn/scale/').replace('.ls2.', '/mlp/scale/')

    n = n.replace('patch_embed.proj.', 'patch/embed/')
    n = n.replace('cls_token', 'patch/cls/token')
    n = n.replace('pos_embed', 'patch/pos/embedding')
    n = n.replace('norm.weight', 'head/norm/gamma').replace('norm.bias', 'head/norm/beta')
    n = n.replace('/weight', '/kernel').replace('.weight', '/kernel').replace('.bias', '/bias')

    n = 'head/proj/kernel:0' if 'weight:0' == n else n
    n = 'head/proj/bias:0' if 'bias:0' == n else n

    return n


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='DinoV2 weight conversion from PyTorch to TensorFlow')
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
    embed_dim, num_heads = model.get_layer(name='layer_0/attn/mhsa')._query_dense.kernel.shape[:2]

    path, weights_torch = 'wrong_path_linear_head.pth', {}
    for checkpoint in CHECKPOINTS[argv.model_type]:
        path = get_file(fname=None, origin=checkpoint, cache_subdir='', cache_dir=argv.out_path)
        weights_torch.update(torch.load(path, map_location='cpu'))
    weights_torch = {convert_name(k): v.numpy() for k, v in weights_torch.items()}
    weights_torch = transform_weights(weights_torch, embed_dim, num_heads)

    weights_tf = []
    for w in model.weights:
        assert w.name in weights_torch, f'Can\'t find weight {w.name} in checkpoint'

        weight = weights_torch[w.name]
        assert w.shape == weight.shape, f'Weight {w.name} shapes not compatible: {w.shape} vs {weight.shape}'

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(path.replace('_linear_head.pth', '_pretrain_linear_head.h5'), save_format='h5')
