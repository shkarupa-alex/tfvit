import tensorflow as tf
from keras.src import testing
from tfvit.ape import AbsolutePositionEmbedding


class TestAbsolutePositionEmbedding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            AbsolutePositionEmbedding,
            init_kwargs={'patch_size': 16, 'pretrain_size': 224, 'num_registers': 0},
            input_shape=(2, 14 ** 2 + 1, 8),
            input_dtype='float32',
            expected_output_shape=(2, 14 ** 2 + 1, 8),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            AbsolutePositionEmbedding,
            init_kwargs={'patch_size': 16, 'pretrain_size': 224, 'num_registers': 2},
            input_shape=(2, 14 ** 2 + 3, 8),
            input_dtype='float32',
            expected_output_shape=(2, 14 ** 2 + 3, 8),
            expected_output_dtype='float32'
        )

    def test_resize(self):
        layer224 = AbsolutePositionEmbedding(32, 224)
        layer224.build([None, 7 ** 2 + 1, 16])

        layer384 = AbsolutePositionEmbedding(32, 384)
        layer384.build([None, 12 ** 2 + 1, 16])
        layer384.set_weights(layer224.get_weights())

        # With registers
        layer224 = AbsolutePositionEmbedding(32, 224, num_registers=2)
        layer224.build([None, 7 ** 2 + 3, 16])

        layer384 = AbsolutePositionEmbedding(32, 384, num_registers=2)
        layer384.build([None, 12 ** 2 + 3, 16])
        layer384.set_weights(layer224.get_weights())


if __name__ == '__main__':
    tf.test.main()
