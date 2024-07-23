import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.src import testing
from keras.src.dtype_policies import dtype_policy
from tfvit.model import ViTTiny16224


class TestModel(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super(TestModel, self).setUp()
        self.default_policy = dtype_policy.dtype_policy()

    def tearDown(self):
        super(TestModel, self).tearDown()
        dtype_policy.set_dtype_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            dtype_policy.set_dtype_policy('mixed_float16')

        model = ViTTiny16224(weights=None)
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

    def test_var_shape(self):
        model = ViTTiny16224(weights=None, include_top=False, input_shape=(None, None, 3))
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 14, 14, 192)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()


if __name__ == '__main__':
    tf.test.main()
