import pytest
from keras.src import testing

from tfvit.clstok import AddClassToken
from tfvit.clstok import SplitClassToken


class TestAddClassToken(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            AddClassToken,
            init_kwargs={"num_registers": 0},
            input_shape=(2, 12, 4),
            input_dtype="float32",
            expected_output_shape=(2, 13, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            AddClassToken,
            init_kwargs={"num_registers": 2},
            input_shape=(2, 12, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 4),
            expected_output_dtype="float32",
        )


class TestSplitClassToken(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            SplitClassToken,
            init_kwargs={
                "patch_size": 32,
                "current_size": 224,
                "num_registers": 0,
            },
            input_shape=(2, 7**2 + 1, 8),
            input_dtype="float32",
            expected_output_shape=((2, 8), (2, 7, 7, 8)),
            expected_output_dtype=("float32", "float32"),
        )
        self.run_layer_test(
            SplitClassToken,
            init_kwargs={
                "patch_size": 32,
                "current_size": 224,
                "num_registers": 2,
            },
            input_shape=(2, 7**2 + 3, 8),
            input_dtype="float32",
            expected_output_shape=((2, 8), (2, 7, 7, 8)),
            expected_output_dtype=("float32", "float32"),
        )
