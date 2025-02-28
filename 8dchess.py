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

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = (2**total_cube_size) * dimension  # Each shift is now 1 byte
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, 0) #Pad with 0 instead of "0" (integers)
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return key_temp
    
    key = pad_or_truncate_key(key_hex)

    arbitrary_key = []
    for cube_coordinate in range(2**total_cube_size):
        key_shifts = []
        for dim in range(dimension):
            key_index = cube_coordinate * dimension + dim
            shift_amount = key[key_index] # Directly use byte value from key
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

def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, cube_data):  #Removed bit_shift_size
    cube_side_length = 2 ** (total_cube_size // dimension)

    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = (2**total_cube_size) * dimension # Each shift is now 1 byte
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, 0) #Pad with 0 instead of "0" (integers)
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      else:
        key_temp = key_hex
      return key_temp
    
    key = pad_or_truncate_key(key_hex) #No need to join string anymore


    arbitrary_key = []
    for cube_coordinate in range(2**total_cube_size):
        key_shifts = []
        for dim in range(dimension):
            key_index = cube_coordinate * dimension + dim
            shift_amount = (cube_side_length - key[key_index]) % cube_side_length  # Directly get byte value
            key_shifts.append(shift_amount)
        arbitrary_key.append(key_shifts)

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

def main():
    #power of 2 params
    byte_data = bytearray([random.randint(0, 255) for _ in range(256)]) # Random bytearray

    dimension = 3
    total_cube_size = 18#2^3
    cube_data = list(byte_data) #total 2^3 = 8 entries,  Keep as byte data

    # print(cube_data)
    if len(cube_data) < 2**(total_cube_size):
        cube_data.extend([0] * (2**(total_cube_size) - len(cube_data)))


    if len(cube_data) > 2**(total_cube_size):
        cube_data = cube_data[0:2**(total_cube_size)]


    #Generate a bytearray for key shifts with the length of (2**total_cube_size)*dimension (one byte per key)
    key_byte = bytearray([random.randint(0, 255) for i in range((2**total_cube_size)*dimension)]) # Bytes


    hypercube, shifted_hypercube = create_hypercube_and_shift(
        dimension, total_cube_size, key_byte, cube_data  #Removed bit_shift_size
    )

    hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
        dimension, total_cube_size, key_byte, shifted_hypercube #Removed bit_shift_size
    )


    print("diff:")
    print(np.sum(np.abs(np.array(hypercube)-np.array(og_hypercube))))

if __name__ == "__main__":
    # start = time.time()
    cProfile.run('main()', 'profile_output')

    p = pstats.Stats('profile_output')
    p.sort_stats('cumulative').print_stats(20)

    # end = time.time()
    # print(end - start)