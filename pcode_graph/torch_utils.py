import torch

class BinaryConverter:
    """Helper to convert between scalar tensor and tensor of bits"""

    def __init__(self, num_bits: int) -> None:
        self.num_bits = num_bits
        self._min_value = -(2 ** (num_bits - 1))
        self._max_value = 2 ** (num_bits - 1) - 1
        self._mask = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.long)

    def value_to_bits(self, value: int) -> torch.Tensor:
        """Converts the given signed value to bits"""

        if value < self._min_value:
            value = self._min_value
        elif value > self._max_value:
            value = self._max_value

        return (
            torch.tensor(value, dtype=torch.long)
            .unsqueeze(-1)
            .bitwise_and(self._mask)
            .ne(0)
            .float()
        )

    def bits_to_value(self, logits: torch.Tensor) -> torch.Tensor:
        bits = (logits > 0).to(torch.long)
        return torch.sum(self._mask * bits, -1)
