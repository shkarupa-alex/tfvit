# tfvit

Keras (TensorFlow v2) reimplementation of **Vision Transformer** model.

+ Based on [Official implementation](https://github.com/google-research/vision_transformer).
+ Contains pretrained weights converted from official ones.
+ Contains pretrained weights converted from [DinoV2+Registers](https://github.com/facebookresearch/dinov2).

## Installation

```bash
pip install tfvit
```

## Examples

Default usage (without preprocessing):

```python
from tfvit import ViTBase32384  # + 11 other variants and input preprocessing

model = ViTBase32384()  # by default will download imagenet[21k]-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfvit import ViTBase32224

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = ViTBase32224(include_top=False)(inputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Differences

Code simplification:

- Pretrain input height and width are always equal
- Patch height and width are always equal

## Citation

```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```