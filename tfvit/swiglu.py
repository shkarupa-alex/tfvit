import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package='TFViT')
class SwiGLU(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        outputs = tf.nn.silu(x1) * x2

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)
