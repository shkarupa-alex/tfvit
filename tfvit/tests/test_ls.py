import tensorflow as tf
from keras.src import testing
from tfvit.ls import LayerScale


class TestLayerScale(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            LayerScale,
            init_kwargs={},
            input_shape=(2, 14 ** 2 + 1, 8),
            input_dtype='float32',
            expected_output_shape=(2, 14 ** 2 + 1, 8),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
