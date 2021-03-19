"""
Contains custom `Faker` providers.
"""


from typing import Any, Iterable, Reversible

import numpy as np
import tensorflow as tf
from faker import Faker
from faker.providers import BaseProvider


class TensorProvider(BaseProvider):
    """
    Provider for creating random Tensors.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.__faker = Faker()

    def tensor(self, shape: Reversible[int]) -> tf.Tensor:
        """
        Creates a fake tensor with arbitrary values and the given shape.

        Args:
            shape: The shape that the tensor should have.

        Returns:
            The tensor that it created.

        """
        # Build up the tensor from the last dimension inward.
        reverse_dimensions = list(reversed(shape))

        tensor = tf.linspace(
            0.0, self.__faker.pyfloat(), reverse_dimensions[0]
        )
        for dim_size in reverse_dimensions[1:]:
            tensor = tf.linspace(
                tensor, tensor + self.__faker.pyfloat(), dim_size
            )

        return tensor

    def ragged_tensor(
        self, *, row_lengths: Iterable[int], inner_shape=Iterable[int]
    ) -> tf.RaggedTensor:
        """
        Creates a fake `RaggedTensor` with arbitrary values and the given shape.

        Args:
            row_lengths: The lengths to use for each row in the tensor.
            inner_shape: The fixed shape to use for each inner element.

        Returns:
            The `RaggedTensor` that it created. The final bounding shape will be
            `[len(row_lengths), max(*row_lengths), *inner_shape]`, where the
            second dimension is ragged.

        """
        # Create the tensor elements.
        num_elements = np.sum(row_lengths)
        elements = self.tensor((num_elements,) + inner_shape)

        # Convert to a `RaggedTensor`.
        return tf.RaggedTensor.from_row_lengths(elements, row_lengths)
