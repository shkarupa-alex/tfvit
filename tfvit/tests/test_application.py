import numpy as np
import tensorflow as tf
import tfvit
from absl.testing import parameterized
from keras.src import layers, models
from keras.src.applications import imagenet_utils
from keras.src.utils import get_file, image_utils

MODEL_LIST = [
    (tfvit.ViTTiny16224, 224, 192),
    (tfvit.ViTTiny16384, 384, 192),

    (tfvit.ViTSmall16224, 224, 384),
    (tfvit.ViTSmall16384, 384, 384),
    (tfvit.ViTSmall32224, 224, 384),
    (tfvit.ViTSmall32384, 384, 384),

    (tfvit.ViTBase16224, 224, 768),
    (tfvit.ViTBase16384, 384, 768),
    (tfvit.ViTBase32224, 224, 768),
    (tfvit.ViTBase32384, 384, 768),

    (tfvit.ViTLarge16224, 224, 1024),
    (tfvit.ViTLarge16384, 384, 1024),

    (tfvit.ViTSmall14518, 518, 384),
    (tfvit.ViTBase14518, 518, 768),
    (tfvit.ViTLarge14518, 518, 1024),
    (tfvit.ViTGiant14518, 518, 1536),
]


class ApplicationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, *_):
        # Can be instantiated with default arguments
        model = app(weights=None)

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)

        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, _, last_dim):
        output_shape = app(weights=None, include_top=False).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_weights_notop(self, app, size, last_dim):
        model = app(include_top=False)
        self.assertEqual(model.output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_predict(self, app, size, _):
        model = app()
        if 1000 != model.output_shape[-1]:
            self.skipTest('Not an IN1k pretrained application.')

        test_image = get_file(
            'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras.src-applications/tests/elephant.jpg')
        image = image_utils.load_img(test_image, target_size=(size, size), interpolation='bicubic')
        image = image_utils.img_to_array(image)[None, ...]

        preds = model.predict(image)

        names = [p[1] for p in imagenet_utils.decode_predictions(preds, top=3)[0]]
        self.assertIn('African_elephant', names)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_backbone(self, app, size, _):
        inputs = layers.Input(shape=(None, None, 3), dtype='uint8')
        outputs = app(include_top=False)(inputs)
        outputs = layers.Conv2D(4, 3, padding='same', activation='softmax')(outputs)
        model = models.Model(inputs=inputs, outputs=outputs)

        data = np.random.uniform(0., 255., size=(2, size, size, 3)).astype('uint8')
        result = model.predict(data)

        self.assertEqual(result.shape[0], 2)
        self.assertIn(result.shape[1], [size // 14, size // 16, size // 32])
        self.assertIn(result.shape[2], [size // 14, size // 16, size // 32])
        self.assertEqual(result.shape[3], 4)


if __name__ == '__main__':
    tf.test.main()
