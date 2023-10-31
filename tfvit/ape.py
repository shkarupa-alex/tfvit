import tensorflow as tf
from keras import initializers, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFViT')
class AbsolutePositionEmbedding(layers.Layer):
    def __init__(self, patch_size, pretrain_size, num_registers=0, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.patch_size = patch_size
        self.pretrain_size = pretrain_size
        self.num_registers = num_registers

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        current_patches = (self.pretrain_size // self.patch_size) ** 2 + 1 + self.num_registers

        # noinspection PyAttributeOutsideInit
        self.embedding = self.add_weight(
            'embedding', shape=[1, current_patches, channels],
            initializer=initializers.TruncatedNormal(stddev=0.02))

        super().build(input_shape)

    def set_weights(self, weights):
        if self.built:
            if 1 != len(weights):
                raise ValueError(
                    f'You called `set_weights(weights)` on layer "{self.name}" with a weight list of length '
                    f'{len(weights)}, but the layer was expecting 1 weight. Provided weights: {weights}')

            if 3 != len(weights[0].shape):
                raise ValueError(
                    f'Layer {self.name} weight shape {self.embedding.shape} is not compatible with provided '
                    f'weight shape {weights[0].shape}.'
                )

            current_size = int((weights[0].shape[1] - 1 - self.num_registers) ** 0.5 * self.patch_size)
            if self.pretrain_size != current_size:
                tf.get_logger().info(
                    f'Resizing absolute position embeddings from {current_size} to {self.pretrain_size}')
                pretrain_patches = self.pretrain_size // self.patch_size
                current_patches = current_size // self.patch_size

                cls_embed, pos_embed = weights[0][:, :1 + self.num_registers], weights[0][:, 1 + self.num_registers:]
                pos_embed = tf.reshape(pos_embed, [1, current_patches, current_patches, -1])
                pos_embed = tf.image.resize(
                    pos_embed, [pretrain_patches, pretrain_patches], method=tf.image.ResizeMethod.BICUBIC)
                pos_embed = tf.reshape(pos_embed, [1, pretrain_patches ** 2, -1])

                weights = [tf.concat([cls_embed, pos_embed], axis=1)]

        super().set_weights(weights)

    def call(self, inputs, *args, **kwargs):
        outputs = inputs + self.embedding

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'pretrain_size': self.pretrain_size,
            'num_registers': self.num_registers
        })

        return config
