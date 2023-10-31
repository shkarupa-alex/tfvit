import numpy as np
import tensorflow as tf
from keras import backend, initializers, layers, mixed_precision, models
from keras.src.applications import imagenet_utils
from keras.src.utils import data_utils, layer_utils
from tfvit.ape import AbsolutePositionEmbedding
from tfvit.clstok import AddClassToken, SplitClassToken

BASE_URL = 'https://github.com/shkarupa-alex/tfvit/releases/download/{}/vit{}_{}.h5'
WEIGHT_URLS = {
    'vit_tiny_16__imagenet': BASE_URL.format('3.0.0', '', 'tiny_patch4_window7_224_22k'),
}
WEIGHT_HASHES = {
    'vit_tiny_16__imagenet': '06588c1314ec16f5d46b6381431c1fc9355f82c4eda0f34e7e57ee32048b6d9a',
}


# TODO: ln_epsilon=1.001e-5
def ViT(
        patch_size, hidden_size, num_layers, num_heads, patch_bias=True, ln_epsilon=1e-06, drop_rate=0.,
        mlp_act='gelu', mlp_ratio=4, img_size=224, img_scale=(1., 0.), img_mean=None, img_std=None, model_name='vit',
        include_top=True, weights=None, input_tensor=None, input_shape=None, input_dtype=None, classes=1000,
        classifier_activation='softmax'):
    """Instantiates the ViT architecture.

    Args:
      patch_size: size used to divide input image.
      hidden_size: patch embedding dimension.
      num_layers: number of transformer layers.
      num_heads: number of attention heads.
      patch_bias: whether to use bias in patch embedding.
      ln_epsilon: layer normalization epsilon constant.
      drop_rate: dropout ratio.
      mlp_act: mlp activation.
      mlp_ratio: mlp expansion ratio.
      img_size: image height and width.
      img_scale: image rescaling constants.
      img_mean: image normalization constants.
      img_std: image normalization constants.
      model_name: model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization), 'imagenet' (pre-training dataset name), or the path to the
        weights file to be loaded.
      input_tensor: tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: shape tuple without batch dimension. Used to create input layer if `input_tensor` not provided.
      input_dtype: input data type. Used to create input layer if `input_tensor` not provided.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', 'imagenet21k', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), '
                         '`imagenet`/`imagenet21k` (pre-training on ImageNet[21k]), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet21k' and include_top and 21843 != classes:
        raise ValueError('If using `weights` as `"imagenet21k"` with `include_top` as true, '
                         '`classes` should equals to 21843.')
    if weights == 'imagenet' and include_top and 1000 != classes:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should equals to 1000.')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape and type
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape, default_size=img_size, min_size=patch_size, data_format='channel_last', require_flatten=False,
        weights=weights)

    if input_dtype is None:
        if (1., 0.) == img_scale and img_mean is None and img_std is None:
            input_dtype = mixed_precision.global_policy().compute_dtype
        else:
            input_dtype = 'uint8'

    # Define model inputs
    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype=input_dtype, name='images')
    else:
        image = layers.Input(shape=input_shape, dtype=input_dtype, name='images')

    # Define model pipeline
    x = image

    if (1., 0.) != img_scale:
        x = layers.Rescaling(scale=img_scale[0], offset=img_scale[1], name='image/scale')(x)
    if not (img_mean is None and img_std is None):
        x = layers.Normalization(mean=img_mean, variance=np.array(img_std) ** 2, name='image/norm')(x)

    x = layers.Conv2D(hidden_size, patch_size, strides=patch_size, use_bias=patch_bias, name='patch/embed')(x)
    x = layers.Reshape([(img_size // patch_size) ** 2, hidden_size], name='patch/flatten')(x)
    x = AddClassToken(name='patch/cls')(x)
    x = AbsolutePositionEmbedding(patch_size, img_size, name='patch/pos')(x)
    x = layers.Dropout(drop_rate, name='patch/drop')(x)

    for i in range(num_layers):
        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'layer_{i}/attn/norm')(x)
        y = layers.MultiHeadAttention(
            num_heads, hidden_size // num_heads, name=f'layer_{i}/attn/mhsa')(y, y)
        y = layers.Dropout(drop_rate, name=f'layer_{i}/attn/drop')(y)
        x = layers.add([x, y], name=f'layer_{i}/attn/add')

        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'layer_{i}/mlp/norm')(x)
        y = layers.Dense(int(hidden_size * mlp_ratio), name=f'layer_{i}/mlp/expand')(y)
        y = layers.Activation(mlp_act, name=f'layer_{i}/mlp/act')(y)
        y = layers.Dense(hidden_size, name=f'layer_{i}/mlp/squeeze')(y)
        y = layers.Dropout(drop_rate, name=f'layer_{i}/layer_{i}/mlp/drop')(y)
        x = layers.add([x, y], name=f'layer_{i}/mlp/add')

    x = layers.LayerNormalization(epsilon=ln_epsilon, name=f'head/norm')(x)
    x, features = SplitClassToken(patch_size, img_size, name='head/split')(x)

    if include_top:
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, name='head/proj')(x)
        x = layers.Activation(classifier_activation, dtype='float32', name='head/act')(x)
    else:
        x = features

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[model_name]
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfvit')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ViT224In21k(img_scale=(2. / 255., -1), classes=21843, **kwargs):
    return ViT(img_scale=img_scale, classes=classes, **kwargs)


def ViT384In1k(img_size=384, img_scale=(2. / 255., -1), **kwargs):
    return ViT(img_size=img_size, img_scale=img_scale, **kwargs)


def ViTTiny16224(
        model_name='vit_tiny_16_224', patch_size=16, hidden_size=192, num_layers=12, num_heads=3, weights='imagenet',
        **kwargs):
    return ViT224In21k(
        model_name=model_name, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers,
        num_heads=num_heads, weights=weights, **kwargs)


def ViTTiny16384(
        model_name='vit_tiny_16_384', patch_size=16, hidden_size=192, num_layers=12, num_heads=3, weights='imagenet',
        **kwargs):
    return ViT384In1k(
        model_name=model_name, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers,
        num_heads=num_heads, weights=weights, **kwargs)

# def ViTSmall16224(
#         model_name='vit_small_16', patch_size=16, hidden_size=384, num_layers=12, num_heads=6, weights='imagenet',
#         **kwargs):
#     return ViT(
#         model_name=model_name, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers,
#         num_heads=num_heads, weights=weights, **kwargs)
#
#
# def ViTSmall32224(model_name='vit_small_32', patch_size=32, **kwargs):
#     return ViTSmall16224(model_name=model_name, patch_size=patch_size, **kwargs)
#
#
# def ViTBase16224(
#         model_name='vit_base_16', patch_size=16, hidden_size=768, num_layers=12, num_heads=12, weights='imagenet',
#         **kwargs):
#     return ViT(
#         model_name=model_name, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers,
#         num_heads=num_heads, weights=weights, **kwargs)
#
#
# def ViTBase32224(model_name='vit_base_32', patch_size=32, **kwargs):
#     return ViTBase16224(model_name=model_name, patch_size=patch_size, **kwargs)
#
#
# def ViTLarge16224(
#         model_name='vit_large_16', patch_size=16, hidden_size=1024, num_layers=24, num_heads=16, drop_rate=0.1,
#         weights='imagenet', **kwargs):
#     return ViT(
#         model_name=model_name, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers,
#         num_heads=num_heads, drop_rate=drop_rate, weights=weights, **kwargs)
