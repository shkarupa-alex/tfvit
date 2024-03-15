import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations, test_utils
from tfvit.swiglu import SwiGLU


@test_combinations.run_all_keras_modes
class TestSwiGLU(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SwiGLU,
            kwargs={},
            input_shape=[2, 14 ** 2 + 1, 8],
            input_dtype='float32',
            expected_output_shape=[None, 14 ** 2 + 1, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
