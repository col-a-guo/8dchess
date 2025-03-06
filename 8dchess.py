import sys
import random
import numpy as np
import time
import cProfile
import pstats
#from bitarray import bitarray  # No longer needed

def pad_right_list(lst, length, value=None):
    if len(lst) >= length:
        return lst
    return lst + [value] * (length - len(lst))

def create_hypercube_and_shift(dimension, big_cube_size, small_cube_side_power, key_hex, cube_data):  #Removed bit_shift_size
    """
    Takes a bytearray as input, transforms it into a hypercube, applies shifts,
    and returns a bytearray.
    """
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
    # small cube of bytes within each cell.
    hypercube_shape = (cube_side_length,) * dimension + (small_cube_side_length,) * dimension

    # Ensure that the cube_data (bytearray) is converted into a numpy array of
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

    # Flatten the hypercube back into a bytearray
    shifted_cube_data = bytearray(shifted_hypercube.flatten().tolist())  # Convert to list for bytearray

    return shifted_cube_data


#############################################
def iterate_ndindex_backwards_generator(shape):
    shape = tuple(shape)
    total_size = np.prod(shape, dtype=np.intp)
    for i in range(total_size - 1, -1, -1):
        index = np.unravel_index(i, shape)
        yield index

def reverse_hypercube_and_reverse_shift(dimension, big_cube_size, small_cube_side_power, key_hex, cube_data):  #Removed bit_shift_size
    """
    Takes a bytearray as input, transforms it into a hypercube, applies reverse shifts,
    and returns a bytearray.
    """
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
    # small cube of bytes within each cell.
    hypercube_shape = (cube_side_length,) * dimension + (small_cube_side_length,) * dimension

    # Ensure that the cube_data (bytearray) is converted into a numpy array of
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
        # Mod the shift amount here, with the correct shift
        shift_amount = ((2**small_cube_side_power) - arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-dimension])][shift_dimension]) % (2**small_cube_side_power)
        shifted_hypercube = roll_axis_slice(shifted_hypercube, shift_amount, shift_dimension) #Use roll_axis_slice
        return shifted_hypercube

    def rotate_small_cube(small_cube, shift_amount, axis):
        """Rotates a single small cube."""
        # Mod the shift amount here
        shift_amount = shift_amount % (2**small_cube_side_power)
        return roll_axis_slice(small_cube, shift_amount, axis) #Use roll_axis_slice

    shifted_hypercube = hypercube.copy()

    # Iterate over all the *outer cube* cells in reverse order
    for outer_cube_index in iterate_ndindex_backwards_generator(hypercube.shape[:-dimension]):
        # Extract the current small cube
        small_cube = hypercube[outer_cube_index]

        # Apply the *reverse* shift to the small cube along each of its dimensions.
        for shift_dimension in iterate_ndindex_backwards_generator(range(dimension)):
            # Use the *same* key for the small cube shift, but reverse the shift
            #Mod the shift amount here
            shift_amount = ((2**small_cube_side_power) - arbitrary_key[np.ravel_multi_index(outer_cube_index, hypercube.shape[:-dimension])][shift_dimension]) % (2**small_cube_side_power)
            small_cube = rotate_small_cube(small_cube, shift_amount, shift_dimension)

        # Replace the original small cube with the shifted version.
        shifted_hypercube[outer_cube_index] = small_cube

    # Outer Shift
    for line_index in iterate_ndindex_backwards_generator(hypercube.shape[:-dimension]):
        for shift_dimension in iterate_ndindex_backwards_generator(range(dimension)):
            shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, tuple(list(line_index) + [0] * dimension))

    # Flatten the hypercube back into a bytearray
    og_cube_data = bytearray(shifted_hypercube.flatten().tolist())

    return og_cube_data

def main():
    
    # power of 2 params
    dimension = 3
    big_cube_side_power = 4
    small_cube_side_power = 3
    big_cube_size = 3*big_cube_side_power # changed from 3*big_cube_side_power
    # Account for 512 bits arranged in 8x8x8 cube
    
    big_cube_side_length = 2**big_cube_side_power

    small_cube_side_length = 2**small_cube_side_power
    #num_bits = 2**big_cube_size * small_cube_side_length**dimension
    num_bytes = big_cube_side_length**dimension * small_cube_side_length**dimension #Correct calculation in bytes


    #bit_data = bitarray()
    #bit_data.extend(random.choice([False, True]) for _ in range(num_bits))

    #cube_data = bit_data  #

    cube_data = bytearray(random.randint(0, 255) for _ in range(num_bytes))

    if len(cube_data) < num_bytes:
        cube_data.extend(bytearray([0] * (num_bytes - len(cube_data)))) #Padded with bytes of 0

    if len(cube_data) > num_bytes:
        cube_data = cube_data[:num_bytes]

    # Generate a bytearray for key shifts with the length of
    # (2**big_cube_size)*dimension (one byte per key)
    key_byte = bytearray([random.randint(0, 64) for i in range((2**big_cube_size) * dimension)])  # Bytes

    # Mod the key bytes *before* creating the hypercube
    key_byte = bytearray([k % (2**small_cube_side_power) for k in key_byte])

    shifted_cube_data = create_hypercube_and_shift(dimension, big_cube_size, small_cube_side_power, key_byte, cube_data)

    og_cube_data = reverse_hypercube_and_reverse_shift(dimension, big_cube_size, small_cube_side_power, key_byte, shifted_cube_data)


    print("diff:")
    print((np.array(cube_data)-np.array(og_cube_data)))

if __name__ == "__main__":
    # start = time.time()
    cProfile.run('main()', 'profile_output')

    p = pstats.Stats('profile_output')
    p.sort_stats('cumulative').print_stats(20)

    # end = time.time()
    # print(end - start)