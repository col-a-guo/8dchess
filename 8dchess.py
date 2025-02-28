import sys
import random
import numpy as np
import time
import cProfile
import pstats

def join_string_list(hex_string):
    return("".join(hex_string))

def pad_right_list(lst, length, value=None):
    if len(lst) >= length:
        return lst
    return lst + [value] * (length - len(lst))

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size))
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, "0")
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return(key_temp)
    
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str = join_string_list(pad_or_truncate_key(key_hex))
    key_str = pad_right_list(key_str, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")

    arbitrary_key = []
    for cube_coordinate in range(2**total_cube_size):
        key_shifts = []
        for dim in range(dimension):
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size)
            dimension_offset = dim*(2**bit_shift_size)

            # Store shift amount as integer directly
            shift_amount = int(key_str[cube_offset+dimension_offset:cube_offset+dimension_offset+(2**bit_shift_size)], 2)
            key_shifts.append(shift_amount)
        arbitrary_key.append(key_shifts)

    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)

    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        shifted_hypercube = hypercube.copy()
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]

        # Create a tuple of slices to select the line along the specified dimension
        slices = [slice(None)] * hypercube.ndim
        slices[shift_dimension] = line_index[shift_dimension] #Fixed: indexing the list of tuples here


        shifted_hypercube = np.roll(shifted_hypercube, shift_amount, axis=shift_dimension)

        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()

    for line_index in np.ndindex(*hypercube.shape):

      for shift_dimension in range(dimension):
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))

    return hypercube, shifted_hypercube

#############################################
def iterate_ndindex_backwards_generator(shape):
    shape = tuple(shape)
    total_size = np.prod(shape, dtype=np.intp)
    for i in range(total_size - 1, -1, -1):
        index = np.unravel_index(i, shape)
        yield index

def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size))
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, "0")
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return(key_temp)

    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str_to_flip = join_string_list(pad_or_truncate_key(key_hex))
    key_str_to_flip = pad_right_list(key_str_to_flip, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")

    arbitrary_key = []
    for cube_coordinate in range(2**total_cube_size):
        key_shifts = []
        for dim in range(dimension):
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size)
            dimension_offset = dim*(2**bit_shift_size)

            # Store shift amount as integer directly
            bits_to_check = key_str_to_flip[cube_offset+dimension_offset:cube_offset+dimension_offset+(2**bit_shift_size)]
            new_int_shift = (cube_side_length - int(bits_to_check, 2)) % cube_side_length
            key_shifts.append(new_int_shift)

    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)

    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        shifted_hypercube = hypercube.copy()
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]

        slices = [slice(None)] * dimension # make a list of ":" slices
        slices[shift_dimension] = line_index[shift_dimension] #and then index the list here.

        shifted_hypercube = np.roll(shifted_hypercube, shift_amount, axis=shift_dimension)

        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    

    for line_index in iterate_ndindex_backwards_generator(hypercube.shape):
      for shift_dimension in iterate_ndindex_backwards_generator(range(dimension)):
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))

    return hypercube, shifted_hypercube

def bytearray_to_bits_array(byte_data):
    return [int(bit) for byte in byte_data for bit in format(byte, '08b')]

def main():
    #power of 2 params
    byte_data = bytearray([random.randint(0, 255) for _ in range(256)]) # Random bytearray

    dimension = 3
    total_cube_size = 15#2^3
    bit_shift_size = 2 #2^2 shifting by 0 to 3
    cube_data = bytearray_to_bits_array(byte_data) #total 2^3 = 8 entries 

    # print(cube_data)
    if len(cube_data) < 2**(total_cube_size):
        cube_data.extend([0] * (2**(total_cube_size) - len(cube_data)))


    if len(cube_data) > 2**(total_cube_size):
        cube_data = cube_data[0:2**(total_cube_size)]

    key_byte = ["".join([str(random.randint(0,1)) for i in range(1)]) for i in range((2**(total_cube_size))*dimension*(2**bit_shift_size))]


    hypercube, shifted_hypercube = create_hypercube_and_shift(
        dimension, total_cube_size, key_byte, bit_shift_size, cube_data
    )

    hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
        dimension, total_cube_size, key_byte, bit_shift_size, shifted_hypercube
    )


    print("diff:")
    print(hypercube-og_hypercube)

if __name__ == "__main__":
    # start = time.time()
    cProfile.run('main()', 'profile_output')

    p = pstats.Stats('profile_output')
    p.sort_stats('cumulative').print_stats(20)

    # end = time.time()
    # print(end - start)