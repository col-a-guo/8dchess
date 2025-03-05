import sys
import random
import numpy as np
import time
import cProfile
import pstats


def pad_right_list(lst, length, value=None):
    if len(lst) >= length:
        return lst
    return lst + [value] * (length - len(lst))

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = (2**total_cube_size) * dimension + (2**total_cube_size)  # Added size for per cell rotation key
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, 0) #Pad with 0 instead of "0" (integers)
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return key_temp
    
    key = pad_or_truncate_key(key_hex)

    arbitrary_key = np.array(key).reshape((2**total_cube_size, dimension + 1)) #Precompute key shifts as a numpy array.  Added +1 for per-cell key

    # MODIFY THIS LINE:  Account for 64 bytes per cell
    hypercube_shape = (cube_side_length,) * dimension + (64,)
    hypercube = np.array(cube_data, dtype=np.uint8).reshape(hypercube_shape) #Ensure uint8 dtype

    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        # Extract the cube index from the full index, ignoring the last dimension (64 bytes)
        cube_index = line_index[:-1]  # Exclude the last dimension index
        shift_amount = arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-1])][shift_dimension]
        shifted_hypercube = np.roll(hypercube, shift_amount, axis=shift_dimension) #No more copy()
        return shifted_hypercube

    def rotate_bytes_within_cells(hypercube, arbitrary_key):
        """Rotates the bytes within each cell of a hypercube based on key."""

        num_cells = np.prod(hypercube.shape[:-1])
        byte_indices = np.arange(64)
        
        cube_indices = np.moveaxis(np.indices(hypercube.shape[:-1]), 0, -1).reshape(num_cells, -1)
        #Extract per cell rotation values
        per_cell_rotations = []
        for i in range(num_cells):
            per_cell_rotations.append(arbitrary_key[np.ravel_multi_index(cube_indices[i], hypercube.shape[:-1])][dimension])

        rotation_shifts = np.array(per_cell_rotations).reshape(num_cells, 1)

        # Calculate rotated indices with broadcasting and modulo
        rotated_indices = (byte_indices - rotation_shifts) % 64

        # Create multi-dimensional indices for advanced indexing
        multi_indices = np.moveaxis(np.indices(hypercube.shape[:-1]), 0, -1).reshape(num_cells, -1)
        # Create destination array for rotated values
        rotated_cube = np.empty_like(hypercube)

        # Advanced indexing for destination values
        for i in range(num_cells):
            rotated_cube[tuple(multi_indices[i])] = hypercube[tuple(multi_indices[i])][rotated_indices[i]]
        return rotated_cube
    
    shifted_hypercube = hypercube.copy()

    #Correct the loop for all the shift_dimension
    for line_index in np.ndindex(*hypercube.shape[:-1]): # Iterate only over cube indices, NOT the 64 byte cells
        for shift_dimension in range(dimension):
            # Add the 0 index for the byte cell dimension
            shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, tuple(list(line_index)+[0])) #Need to add a 0 for byte offset in order to correctly shift

    shifted_hypercube = rotate_bytes_within_cells(shifted_hypercube, arbitrary_key)

    return hypercube, shifted_hypercube

#############################################
def iterate_ndindex_backwards_generator(shape):
    shape = tuple(shape)
    total_size = np.prod(shape, dtype=np.intp)
    for i in range(total_size - 1, -1, -1):
        index = np.unravel_index(i, shape)
        yield index

def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (total_cube_size // dimension)

    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = (2**total_cube_size) * dimension + (2**total_cube_size)  # Added size for per cell rotation key
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, 0) #Pad with 0 instead of "0" (integers)
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return key_temp
    
    key = pad_or_truncate_key(key_hex)
    arbitrary_key = np.array(key).reshape((2**total_cube_size, dimension + 1)) #Precompute key shifts as a numpy array


    # MODIFY THIS LINE: Account for 64 bytes per cell
    hypercube_shape = (cube_side_length,) * dimension + (64,)
    hypercube = np.array(cube_data, dtype=np.uint8).reshape(hypercube_shape) #Ensure uint8 dtype

    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        cube_index = line_index[:-1]  # Exclude the last dimension index
        shift_amount = (cube_side_length - arbitrary_key[np.ravel_multi_index(cube_index, hypercube.shape[:-1])][shift_dimension]) % cube_side_length #Precompute
        shifted_hypercube = np.roll(hypercube, shift_amount, axis=shift_dimension) #No more copy()
        return shifted_hypercube
    
    def rotate_bytes_within_cells(hypercube, arbitrary_key):
        """Rotates the bytes within each cell of a hypercube based on key."""

        num_cells = np.prod(hypercube.shape[:-1])
        byte_indices = np.arange(64)
        
        cube_indices = np.moveaxis(np.indices(hypercube.shape[:-1]), 0, -1).reshape(num_cells, -1)
        #Extract per cell rotation values
        per_cell_rotations = []
        for i in range(num_cells):
            rotation_val = (64 - arbitrary_key[np.ravel_multi_index(cube_indices[i], hypercube.shape[:-1])][dimension]) % 64
            per_cell_rotations.append(rotation_val)

        rotation_shifts = np.array(per_cell_rotations).reshape(num_cells, 1)

        # Calculate rotated indices with broadcasting and modulo
        rotated_indices = (byte_indices - rotation_shifts) % 64

        # Create multi-dimensional indices for advanced indexing
        multi_indices = np.moveaxis(np.indices(hypercube.shape[:-1]), 0, -1).reshape(num_cells, -1)
        # Create destination array for rotated values
        rotated_cube = np.empty_like(hypercube)

        # Advanced indexing for destination values
        for i in range(num_cells):
            rotated_cube[tuple(multi_indices[i])] = hypercube[tuple(multi_indices[i])][rotated_indices[i]]
        return rotated_cube

    shifted_hypercube = hypercube.copy() #Now cube has uint8 type

    #Correct the loop for all the shift_dimension
    for line_index in iterate_ndindex_backwards_generator(hypercube.shape[:-1]): # Iterate only over cube indices, NOT the 64 byte cells
        for shift_dimension in iterate_ndindex_backwards_generator(range(dimension)):
             shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, tuple(list(line_index)+[0])) #Need to add a 0 for byte offset in order to correctly shift

    shifted_hypercube = rotate_bytes_within_cells(shifted_hypercube, arbitrary_key)

    return hypercube, shifted_hypercube

def main():
    #power of 2 params
    dimension = 3
    total_cube_size = 12#

    # MODIFY THIS LINE:  Account for 64 bytes per cell.  Each cell now has 64 bytes.
    byte_data = bytearray([random.randint(0, 255) for _ in range(2**total_cube_size * 64)]) # Random bytearray
    cube_data = list(byte_data) #

    # print(cube_data)
    if len(cube_data) < 2**(total_cube_size) * 64:
        cube_data.extend([0] * (2**(total_cube_size) * 64 - len(cube_data)))


    if len(cube_data) > 2**(total_cube_size) * 64:
        cube_data = cube_data[0:2**(total_cube_size) * 64]


    #Generate a bytearray for key shifts with the length of (2**total_cube_size)*dimension (one byte per key)
    key_length = (2**total_cube_size)*dimension + (2**total_cube_size) #Bytes
    key_byte = bytearray([random.randint(0, 255) for i in range(key_length)]) # Bytes


    hypercube, shifted_hypercube = create_hypercube_and_shift(
        dimension, total_cube_size, key_byte, cube_data  #Removed bit_shift_size
    )

    hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
        dimension, total_cube_size, key_byte, shifted_hypercube #Removed bit_shift_size
    )


    print("diff:")
    print((np.array(hypercube)-np.array(og_hypercube)))

if __name__ == "__main__":
    # start = time.time()
    cProfile.run('main()', 'profile_output')

    p = pstats.Stats('profile_output')
    p.sort_stats('cumulative').print_stats(20)

    # end = time.time()
    # print(end - start)