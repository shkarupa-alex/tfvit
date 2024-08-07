from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFViT")
class AddClassToken(layers.Layer):
    def __init__(self, num_registers=0, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.num_registers = num_registers

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.token = self.add_weight(
            name="token",
            shape=(1, 1 + self.num_registers, channels),
            initializer="zeros",
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch_size = ops.shape(inputs)[0]
        cls_token = ops.repeat(self.token, batch_size, axis=0)
        outputs = ops.concatenate([cls_token, inputs], axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if length is None:
            return input_shape

        return (
            input_shape[:1]
            + (length + 1 + self.num_registers,)
            + input_shape[2:]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"num_registers": self.num_registers})

        return config


@register_keras_serializable(package="TFViT")
class SplitClassToken(layers.Layer):
    def __init__(self, patch_size, current_size, num_registers=0, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

        self.patch_size = patch_size
        self.current_size = current_size
        self.num_registers = num_registers

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: self.channels})

        # noinspection PyAttributeOutsideInit
        self.features_size = self.current_size // self.patch_size

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        token, _, features = ops.split(
            inputs, [1, 1 + self.num_registers], axis=1
        )
        token = ops.reshape(token, [-1, self.channels])
        features = ops.reshape(
            features,
            [-1, self.features_size, self.features_size, self.channels],
        )

        return token, features

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[:1] + (
            self.features_size,
            self.features_size,
            self.channels,
        )
        token_shape = input_shape[:1] + (self.channels,)

        return token_shape, features_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "current_size": self.current_size,
                "num_registers": self.num_registers,
            }
        )

        return config
