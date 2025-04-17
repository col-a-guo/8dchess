import unittest
import math
from eightDchess import pad_with_pi

class TestPadWithPi(unittest.TestCase):

    def test_pad_with_pi_basic(self):
        data = bytearray([1, 2, 3])
        required_size = 10
        padded_data = pad_with_pi(data, required_size)
        self.assertEqual(len(padded_data), required_size)
        self.assertEqual(padded_data[:3], data)

        # Check that the padding is actually pi digits.
        pi_digits = str(math.pi).replace('.', '')
        expected_padding = bytearray()
        pi_index = 0
        for _ in range(7):  # Need to pad 7 bytes to reach size 10
            digit_pair = pi_digits[pi_index:pi_index + 2]
            if len(digit_pair) == 2:
                try:
                    expected_padding.append(int(digit_pair))
                except ValueError:
                    expected_padding.append(0)
            else:
                expected_padding.append(0)
            pi_index = (pi_index + 2) % len(pi_digits)

        self.assertEqual(padded_data[3:], expected_padding)

if __name__ == '__main__':
    unittest.main()