import unittest
import numpy as np
import math
import random
from eightDchess import pad_with_pi, unpad_with_pi, bitarray_from_bytearray, create_hypercube_of_squares, create_index_cube, apply_rotations_to_index_cube, reverse_rotations_to_index_cube, encrypt_byte_array, decrypt_hypercube

class TestEncryptDecrypt(unittest.TestCase):

    def test_encrypt_decrypt_small(self):
        hypercube_length = 2
        square_length = 2
        num_dimensions = 1
        data_size = hypercube_length**num_dimensions * square_length*square_length
        key_size = (hypercube_length**num_dimensions) * num_dimensions
        original_byte_array = bytearray(random.getrandbits(8) for _ in range(data_size // 8))  # Use smaller data size
        key = bytearray(random.getrandbits(8) for _ in range(key_size)) # getrandbits needs bits, not bytes

        original_byte_array = pad_with_pi(original_byte_array, data_size // 8)

        # now convert the key to integers
        key = [x for x in key]

        key = pad_with_pi(bytearray(key), key_size) #pad to key size
        key = [x for x in key]

        self.assertEqual(len(key), key_size)

        encrypted_cube = encrypt_byte_array(original_byte_array, key, hypercube_length, square_length, num_dimensions)
        decrypted_byte_array = decrypt_hypercube(encrypted_cube, key, hypercube_length, square_length, num_dimensions)

        unpadded_decrypted = unpad_with_pi(decrypted_byte_array)
        unpadded_original = unpad_with_pi(original_byte_array)

        self.assertEqual(unpadded_decrypted, unpadded_original)

    def test_encrypt_decrypt_larger(self):
        hypercube_length = 3
        square_length = 4
        num_dimensions = 2
        data_size = hypercube_length**num_dimensions * square_length*square_length
        key_size = (hypercube_length**num_dimensions) * num_dimensions
        original_byte_array = bytearray(random.getrandbits(8) for _ in range(data_size // 8))  # Use smaller data size
        key = bytearray(random.getrandbits(8) for _ in range(key_size)) # getrandbits needs bits, not bytes

        original_byte_array = pad_with_pi(original_byte_array, data_size // 8)

        # now convert the key to integers
        key = [x for x in key]

        key = pad_with_pi(bytearray(key), key_size) #pad to key size
        key = [x for x in key]


        self.assertEqual(len(key), key_size)

        encrypted_cube = encrypt_byte_array(original_byte_array, key, hypercube_length, square_length, num_dimensions)
        decrypted_byte_array = decrypt_hypercube(encrypted_cube, key, hypercube_length, square_length, num_dimensions)

        unpadded_decrypted = unpad_with_pi(decrypted_byte_array)
        unpadded_original = unpad_with_pi(original_byte_array)

        self.assertEqual(unpadded_decrypted, unpadded_original)
class TestReverseRotationsToIndexCube(unittest.TestCase):

    def test_reverse_rotations_1d(self):
        hypercube_length = 3
        num_dimensions = 1
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        key = [1, 0, 2]
        rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
        reversed_index_cube = reverse_rotations_to_index_cube(rotated_index_cube, key, hypercube_length, num_dimensions)
        np.testing.assert_array_equal(reversed_index_cube, index_cube) # Should be back to original

    def test_reverse_rotations_2d(self):
        hypercube_length = 2
        num_dimensions = 2
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        key = [1, 0, 0, 1, 1, 0, 0, 1]
        rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
        reversed_index_cube = reverse_rotations_to_index_cube(rotated_index_cube, key, hypercube_length, num_dimensions)
        np.testing.assert_array_equal(reversed_index_cube, index_cube) # Should be back to original

    def test_reverse_rotations_no_rotation(self):
        hypercube_length = 2
        num_dimensions = 2
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        key = [0] * (hypercube_length**num_dimensions * num_dimensions)  # All zeros for no rotation
        rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
        reversed_index_cube = reverse_rotations_to_index_cube(rotated_index_cube, key, hypercube_length, num_dimensions)
        np.testing.assert_array_equal(reversed_index_cube, index_cube) # Should be back to original


class TestApplyRotationsToIndexCube(unittest.TestCase):

    def test_apply_rotations_2d(self):
        hypercube_length = 2
        num_dimensions = 2
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        key = [1, 0, 0, 1, 1, 0, 0, 1]  #Key for each cell/dim combination
        rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)

        expected_cube = np.array([[[0,0],[0,1]],[[1,0],[1,1]]])
        np.testing.assert_array_equal(rotated_index_cube, expected_cube)

    def test_apply_rotations_no_rotation(self):
        hypercube_length = 2
        num_dimensions = 2
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        key = [0] * (hypercube_length**num_dimensions * num_dimensions)  # All zeros for no rotation
        rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
        np.testing.assert_array_equal(rotated_index_cube, index_cube)  # Should be identical to original
class TestCreateIndexCube(unittest.TestCase):

    def test_create_index_cube_1d(self):
        hypercube_length = 3
        num_dimensions = 1
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        self.assertEqual(index_cube.shape, (3, 1))
        np.testing.assert_array_equal(index_cube[0], [0])
        np.testing.assert_array_equal(index_cube[1], [1])
        np.testing.assert_array_equal(index_cube[2], [2])

    def test_create_index_cube_2d(self):
        hypercube_length = 2
        num_dimensions = 2
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        self.assertEqual(index_cube.shape, (2, 2, 2))
        np.testing.assert_array_equal(index_cube[0, 0], [0, 0])
        np.testing.assert_array_equal(index_cube[0, 1], [0, 1])
        np.testing.assert_array_equal(index_cube[1, 0], [1, 0])
        np.testing.assert_array_equal(index_cube[1, 1], [1, 1])

    def test_create_index_cube_3d(self):
        hypercube_length = 2
        num_dimensions = 3
        index_cube = create_index_cube(hypercube_length, num_dimensions)
        self.assertEqual(index_cube.shape, (2, 2, 2, 3))
        np.testing.assert_array_equal(index_cube[0, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(index_cube[0, 0, 1], [0, 0, 1])
        np.testing.assert_array_equal(index_cube[1, 1, 1], [1, 1, 1])
class TestCreateHypercubeOfSquares(unittest.TestCase):

    def test_create_hypercube_of_squares_basic(self):
        bitarr = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        hypercube_length = 2
        square_length = 1
        num_dimensions = 1
        reshaped = create_hypercube_of_squares(bitarr, hypercube_length, square_length, num_dimensions)
        self.assertEqual(reshaped.shape, (2, 1, 1))
        self.assertEqual(reshaped[0][0][0], 0)
        self.assertEqual(reshaped[1][0][0], 1)

    def test_create_hypercube_of_squares_2d(self):
        bitarr = np.array(range(8), dtype=np.uint8)  # Use range for more distinct values
        hypercube_length = 2
        square_length = 1
        num_dimensions = 2
        reshaped = create_hypercube_of_squares(bitarr, hypercube_length, square_length, num_dimensions)
        self.assertEqual(reshaped.shape, (2, 2, 1, 1))  # Expected shape for 2D hypercube
        self.assertEqual(reshaped[0][0][0][0], 0)
        self.assertEqual(reshaped[0][1][0][0], 1)
        self.assertEqual(reshaped[1][0][0][0], 2)
        self.assertEqual(reshaped[1][1][0][0], 3)


    def test_create_hypercube_of_squares_not_enough_data(self):
       bitarr = np.array([0, 1, 0, 1], dtype=np.uint8)  # Insufficient data
       hypercube_length = 2
       square_length = 2
       num_dimensions = 1
       with self.assertRaises(ValueError): # Expect a ValueError because not enough data
           create_hypercube_of_squares(bitarr, hypercube_length, square_length, num_dimensions)


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


class TestUnpadWithPi(unittest.TestCase):

    def test_unpad_with_pi_no_padding(self):
        data = bytearray([1, 2, 3, 4, 5])
        unpadded_data = unpad_with_pi(data)
        self.assertEqual(unpadded_data, data)

    def test_unpad_with_pi_empty_data(self):
        data = bytearray()
        unpadded_data = unpad_with_pi(data)
        self.assertEqual(unpadded_data, data)

    def test_unpad_with_pi_partial_padding(self):  #Test case where the pi sequence is shorter than expected
        data = bytearray([1,2,3])
        pi_digits = str(math.pi).replace('.', '')
        partial_pi_digits = pi_digits[:3]
        padding = bytearray()
        for i in range(0, len(partial_pi_digits), 2):
             digit_pair = partial_pi_digits[i:i+2]
             if len(digit_pair) == 2:
                 try:
                     padding.append(int(digit_pair))
                 except ValueError:
                     padding.append(0)
             else:
                 break #stop if can't get two digits
        padded_data = data + padding
        unpadded_data = unpad_with_pi(bytearray(padded_data)) #have to recast to bytearray for indexing reasons
        self.assertEqual(unpadded_data, bytearray(padded_data)) #should be equal because didn't find pi sequence


class TestBitarrayFromBytearray(unittest.TestCase):

    def test_bitarray_from_bytearray_basic(self):
        bytearr = bytearray([10, 20, 30])
        bitarr = bitarray_from_bytearray(bytearr)
        self.assertEqual(len(bitarr), len(bytearr) * 8)

        # Verify the bit representation of each byte
        expected_bits = np.array([
            [0, 0, 0, 0, 1, 0, 1, 0],  # 10
            [0, 0, 0, 1, 0, 1, 0, 0],  # 20
            [0, 0, 0, 1, 1, 1, 1, 0]   # 30
        ]).flatten()

        np.testing.assert_array_equal(bitarr, expected_bits) # Compare numpy arrays

    def test_bitarray_from_bytearray_empty(self):
        bytearr = bytearray()
        bitarr = bitarray_from_bytearray(bytearr)
        self.assertEqual(len(bitarr), 0)

    def test_bitarray_from_bytearray_single_byte(self):
        bytearr = bytearray([255])
        bitarr = bitarray_from_bytearray(bytearr)
        self.assertEqual(len(bitarr), 8)
        expected_bits = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(bitarr, expected_bits)


if __name__ == '__main__':
    unittest.main()