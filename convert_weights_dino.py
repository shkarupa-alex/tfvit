#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from keras.src.utils import get_file

from tfvit import model

BASE_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_{}.pth"
CHECKPOINTS = {
    # https://github.com/facebookresearch/dinov2
    "vit_small_14_518__dino2reg": [
        BASE_URL.format("vits14/dinov2_vits14_reg4_pretrain"),
        BASE_URL.format("vits14/dinov2_vits14_reg4_linear_head"),
    ],
    "vit_base_14_518__dino2reg": [
        BASE_URL.format("vitb14/dinov2_vitb14_reg4_pretrain"),
        BASE_URL.format("vitb14/dinov2_vitb14_reg4_linear_head"),
    ],
    "vit_large_14_518__dino2reg": [
        BASE_URL.format("vitl14/dinov2_vitl14_reg4_pretrain"),
        BASE_URL.format("vitl14/dinov2_vitl14_reg4_linear_head"),
    ],
    "vit_giant_14_518__dino2reg": [
        BASE_URL.format("vitg14/dinov2_vitg14_reg4_pretrain"),
        BASE_URL.format("vitg14/dinov2_vitg14_reg4_linear_head"),
    ],
}
MODELS = {
    "vit_small_14_518__dino2reg": model.ViTSmall14518,
    "vit_base_14_518__dino2reg": model.ViTBase14518,
    "vit_large_14_518__dino2reg": model.ViTLarge14518,
    "vit_giant_14_518__dino2reg": model.ViTGiant14518,
}


def transform_weights(weights, embed, heads):
    head_dim = embed // heads

    weights.pop("mask_token")

    weights["patch_cls/token"] = np.concatenate(
        [weights["patch_cls/token"], weights.pop("register_tokens")], axis=1
    )

    weights["patch_pos/embedding"] = np.concatenate(
        [
            weights["patch_pos/embedding"][:, :1],
            np.zeros([1, 4, embed], "float32"),
            weights["patch_pos/embedding"][:, 1:],
        ],
        axis=1,
    )

    for key in list(weights.keys()):
        value = weights[key]

        if "patch_embed/kernel" in key:
            value = value.transpose(2, 3, 1, 0)

        if "_scale/" in key:
            value = value[None, None]

        if any(
            [
                part in key
                for part in {
                    "/value/",
                    "/attention_output/",
                    "_expand/",
                    "_squeeze/",
                    "_proj/",
                }
            ]
        ):
            value = value.T

        if "/value/bias" in key:
            value = value.reshape(3, heads, head_dim)
            weights[key.replace("/value/", "/query/")] = value[0]
            weights[key.replace("/value/", "/key/")] = value[1]
            value = value[2]

        if "/value/kernel" in key:
            value = value.reshape(embed, 3, heads, head_dim)
            weights[key.replace("/value/", "/query/")] = value[:, 0]
            weights[key.replace("/value/", "/key/")] = value[:, 1]
            value = value[:, 2]

        if "/attention_output/kernel" in key:
            value = value.reshape(heads, head_dim, embed)

        weights[key] = value

    return weights


def convert_name(n):
    n = n.replace("blocks.", "layer_")
    n = n.replace(".norm1.weight", "_attn_norm/gamma").replace(
        ".norm1.bias", "_attn_norm/beta"
    )
    n = n.replace(".norm2.weight", "_mlp_norm/gamma").replace(
        ".norm2.bias", "_mlp_norm/beta"
    )
    n = n.replace(".attn.qkv.", "_attn_mhsa/value/").replace(
        ".attn.proj.", "_attn_mhsa/attention_output/"
    )
    n = n.replace(".mlp.fc1.", "_mlp_expand/").replace(
        ".mlp.fc2.", "_mlp_squeeze/"
    )
    n = n.replace(".mlp.w12.", "_mlp_expand/").replace(
        ".mlp.w3.", "_mlp_squeeze/"
    )
    n = n.replace(".ls1.", "_attn_scale/").replace(".ls2.", "_mlp_scale/")

    n = n.replace("patch_embed.proj.", "patch_embed/")
    n = n.replace("cls_token", "patch_cls/token")
    n = n.replace("pos_embed", "patch_pos/embedding")
    n = n.replace("norm.weight", "head_norm/gamma").replace(
        "norm.bias", "head_norm/beta"
    )
    n = (
        n.replace("/weight", "/kernel")
        .replace(".weight", "/kernel")
        .replace(".bias", "/bias")
    )

    n = "head_proj/kernel" if "weight" == n else n
    n = "head_proj/bias" if "bias" == n else n

    return n


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="DinoV2 weight conversion from PyTorch to Keras"
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "out_path", type=str, help="Path to save Keras model weights"
    )

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(
        argv.out_path
    ), "Wrong output path"

    model = MODELS[argv.model_type](weights=None)
    embed_dim, num_heads = model.get_layer(
        name="layer_0_attn_mhsa"
    )._query_dense.kernel.shape[:2]

    path, weights_torch = "wrong_path_linear_head.pth", {}
    for checkpoint in CHECKPOINTS[argv.model_type]:
        path = get_file(
            fname=None,
            origin=checkpoint,
            cache_subdir="",
            cache_dir=argv.out_path,
        )
        weights_torch.update(torch.load(path, map_location="cpu"))
    weights_torch = {
        convert_name(k): v.numpy() for k, v in weights_torch.items()
    }
    weights_torch = transform_weights(weights_torch, embed_dim, num_heads)

    weights_keras = []
    for w in model.weights:
        assert (
            w.path in weights_torch
        ), f"Can't find weight {w.path} in checkpoint"

        weight = weights_torch[w.path]
        assert (
            w.shape == weight.shape
        ), f"Weight {w.path} shapes not compatible: {w.shape} vs {weight.shape}"

        weights_keras.append(weight)

    model.set_weights(weights_keras)
    model.save_weights(
        path.replace("_linear_head.pth", "_pretrain_linear_head.weights.h5")
    )
