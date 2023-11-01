import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFViT')
class SwiGLU(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        outputs = tf.nn.silu(x1) * x2

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)
