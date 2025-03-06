import unittest
import numpy as np
import unittest.mock as mock
def pad_right_list(lst, length, value=None):
    if len(lst) >= length:
        return lst
    return lst + [value] * (length - len(lst))

def create_hypercube_and_shift(dimension, big_cube_size, small_cube_side_power, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (big_cube_size // dimension)
    small_cube_side_length = 2**small_cube_side_power

    def pad_or_truncate_key(key_hex):
        key_temp = key_hex
        size = (2**big_cube_size) * dimension  # Each shift is now 1 byte
        if len(key_hex) < size:
            key_temp = pad_right_list(key_hex, size, 0)  # Pad with 0 instead of "0" (integers)
        elif len(key_hex) > size:
            key_temp = key_hex[:size]
        else:
            key_temp = key_hex
        return key_temp

    key = pad_or_truncate_key(key_hex)
    arbitrary_key = np.array(key).reshape((2**big_cube_size, dimension))  # Precompute key shifts

    # Calculate the shape of the hypercube. The last dimensions are the
    # small cube of bits within each cell.
    hypercube_shape = (cube_side_length,) * dimension + (small_cube_side_length,) * dimension

    # Ensure that the cube_data (bitarray) is converted into a numpy array of
    # integers (0-255) before reshaping.
    hypercube = np.array(cube_data).reshape(hypercube_shape)

    def roll_axis_slice(arr, shift, axis):
        """Rolls an array along a single axis using slicing."""
        shift = shift % arr.shape[axis]  # Normalize shift
        if shift == 0:
            return arr  # No shift needed
        idx = tuple(slice(None) if i != axis else slice(shift, None) for i in range(arr.ndim))
        idx_rem = tuple(slice(None) if i != axis else slice(0, shift) for i in range(arr.ndim))
        res = np.concatenate((arr[idx], arr[idx_rem]), axis=axis)
        return res

    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        cube_index = line_index[:-dimension]  # Exclude the last dimensions
        # Mod the shift amount here
        shift_amount = arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-dimension])][shift_dimension] % (2**small_cube_side_power)
        shifted_hypercube = roll_axis_slice(hypercube, shift_amount, shift_dimension)  # Use roll_axis_slice
        return shifted_hypercube

    def rotate_small_cube(small_cube, shift_amount, axis):
        """Rotates a single small cube."""
        # Mod the shift amount here
        shift_amount = shift_amount % (2**small_cube_side_power)
        return roll_axis_slice(small_cube, shift_amount, axis) #Use roll_axis_slice

    shifted_hypercube = hypercube.copy()

    # Iterate over all the *outer cube* cells
    for outer_cube_index in np.ndindex(*hypercube.shape[:-dimension]):
        # Extract the current small cube
        small_cube = hypercube[outer_cube_index]

        # Apply the same shift to the small cube along each of its dimensions.

        for shift_dimension in range(dimension):
            # Use the *same* key for the small cube shift
            # Mod the shift amount here
            shift_amount = arbitrary_key[np.ravel_multi_index(outer_cube_index, hypercube.shape[:-dimension])][shift_dimension] % (2**small_cube_side_power)
            small_cube = rotate_small_cube(small_cube, shift_amount, shift_dimension)

        # Replace the original small cube with the shifted version.
        shifted_hypercube[outer_cube_index] = small_cube

    # Outer Shift
    for line_index in np.ndindex(*hypercube.shape[:-dimension]):
        for shift_dimension in range(dimension):
            shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, tuple(list(line_index) + [0] * dimension))

    return hypercube, shifted_hypercube


#############################################
def iterate_ndindex_backwards_generator(*shape):
    """
    Generates indices in reverse order, similar to np.ndindex, but accepts
    shape as separate arguments using the asterisk.
    """
    shape = tuple(shape)  # Convert the arguments into a tuple

    indices = [s - 1 for s in shape] #Start at the end of the dimension

    while True:
      yield tuple(indices)

      i = len(shape) - 1
      while i >= 0:
        indices[i] -= 1 # Subtract from the index
        if indices[i] < 0: #Check against 0
          indices[i] = shape[i] - 1 #reset to end of dimension
          i -= 1
        else:
          break
      else:
        # All dimensions have reached the end, so stop iterating
        return

def reverse_hypercube_and_reverse_shift(dimension, big_cube_size, small_cube_side_power, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (big_cube_size // dimension)
    small_cube_side_length = 2**small_cube_side_power

    def pad_or_truncate_key(key_hex):
        key_temp = key_hex
        size = (2**big_cube_size) * dimension  # Each shift is now 1 byte
        if len(key_hex) < size:
            key_temp = pad_right_list(key_hex, size, 0)  # Pad with 0 instead of "0" (integers)
        elif len(key_hex) > size:
            key_temp = key_hex[:size]
        else:
            key_temp = key_hex
        return key_temp

    key = pad_or_truncate_key(key_hex)
    arbitrary_key = np.array(key).reshape((2**big_cube_size, dimension))  # Precompute key shifts

    # Calculate the shape of the hypercube. The last dimensions are the
    # small cube of bits within each cell.
    hypercube_shape = (cube_side_length,) * dimension + (small_cube_side_length,) * dimension

    # Ensure that the cube_data (bitarray) is converted into a numpy array of
    # integers (0-255) before reshaping.
    hypercube = np.array(cube_data).reshape(hypercube_shape)

    def roll_axis_slice(arr, shift, axis):
        """Rolls an array along a single axis using slicing."""
        shift = shift % arr.shape[axis]  # Normalize shift
        if shift == 0:
            return arr  # No shift needed
        idx = tuple(slice(None) if i != axis else slice(shift, None) for i in range(arr.ndim))
        idx_rem = tuple(slice(None) if i != axis else slice(0, shift) for i in range(arr.ndim))
        res = np.concatenate((arr[idx], arr[idx_rem]), axis=axis)
        return res

    def backwards_rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        cube_index = line_index[:-dimension]  # Exclude the last dimensions
        # Mod the shift amount here
        shift_amount = arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-dimension])][shift_dimension] % (2**small_cube_side_power)
        shifted_hypercube = roll_axis_slice(hypercube, -1*shift_amount, shift_dimension)  # Use roll_axis_slice
        return shifted_hypercube

    def rotate_small_cube(small_cube, shift_amount, axis):
        """Rotates a single small cube."""
        # Mod the shift amount here
        shift_amount = shift_amount % (2**small_cube_side_power)
        return roll_axis_slice(small_cube, shift_amount, axis) #Use roll_axis_slice

    shifted_hypercube = hypercube.copy()
    # Outer Shift first because we're in reverse
    for line_index in  iterate_ndindex_backwards_generator(*hypercube.shape[:-dimension]):
        for shift_dimension in range(dimension-1, -1, -1):
            shifted_hypercube = backwards_rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, tuple(list(line_index) + [0] * dimension))

    # Iterate over all the *outer cube* cells
    for outer_cube_index in iterate_ndindex_backwards_generator(*hypercube.shape[:-dimension]):
        # Extract the current small cube
        small_cube = hypercube[outer_cube_index]

        # Apply the same shift to the small cube along each of its dimensions.
        for shift_dimension in range(dimension-1, -1, -1):
            # Use the *same* key for the small cube shift
            # Mod the shift amount here
            shift_amount = arbitrary_key[np.ravel_multi_index(outer_cube_index, hypercube.shape[:-dimension])][shift_dimension] % (2**small_cube_side_power)
            small_cube = rotate_small_cube(small_cube, -1*shift_amount, shift_dimension) #negative shift

        # Replace the original small cube with the shifted version.
        shifted_hypercube[outer_cube_index] = small_cube


    return hypercube, shifted_hypercube


class TestRotateBitShiftReverse(unittest.TestCase):

    def test_rotate_bit_shift_forward_and_backward(self):
         def roll_axis_slice(arr, shift, axis):
            """Rolls an array along a single axis using slicing."""
            shift = shift % arr.shape[axis]  # Normalize shift
            if shift == 0:
                return arr  # No shift needed
            idx = tuple(slice(None) if i != axis else slice(shift, None) for i in range(arr.ndim))
            idx_rem = tuple(slice(None) if i != axis else slice(0, shift) for i in range(arr.ndim))
            res = np.concatenate((arr[idx], arr[idx_rem]), axis=axis)
            return res

         def rotate_bit_shift_forward(hypercube, arbitrary_key, shift_dimension, line_index, small_cube_side_power):
            cube_index = line_index[:-2]  # Exclude the last dimensions
            dimension = 2

            # Mod the shift amount here
            shift_amount = arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-2])][shift_dimension] % (2**small_cube_side_power)
            shifted_hypercube = roll_axis_slice(hypercube, shift_amount, shift_dimension)  # Use roll_axis_slice
            return shifted_hypercube

         def backwards_rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index, small_cube_side_power):
            cube_index = line_index[:-2]  # Exclude the last dimensions
            dimension = 2
            # Mod the shift amount here
            shift_amount = arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-2])][shift_dimension] % (2**small_cube_side_power)
            shifted_hypercube = roll_axis_slice(hypercube, -1*shift_amount, shift_dimension)  # Use roll_axis_slice
            return shifted_hypercube

         hypercube = np.array([1, 2, 3, 4, 5])
         arbitrary_key = np.array([[1, 0], [0, 1]])  # Example key
         shift_dimension = 0
         line_index = (0, 0)
         small_cube_side_power = 2
         dimension = 2

         # Rotate forward, then backward
         hypercube_shape = (5,)
         arbitrary_key = np.array([[1]])

         #Mocking
         #with mock.patch('your_module.arbitrary_key', return_value=1):
         forward_shifted = rotate_bit_shift_forward(hypercube.copy(), arbitrary_key, shift_dimension, line_index, small_cube_side_power)
         backward_shifted = backwards_rotate_bit_shift(forward_shifted.copy(), arbitrary_key, shift_dimension, line_index, small_cube_side_power)

         np.testing.assert_array_equal(backward_shifted, hypercube)

         # Rotate backward, then forward
         #with mock.patch('your_module.arbitrary_key', return_value=1):
         backward_shifted = backwards_rotate_bit_shift(hypercube.copy(), arbitrary_key, shift_dimension, line_index, small_cube_side_power)
         forward_shifted = rotate_bit_shift_forward(backward_shifted.copy(), arbitrary_key, shift_dimension, line_index, small_cube_side_power)
         np.testing.assert_array_equal(forward_shifted, hypercube)