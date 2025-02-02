import random
import sys
import math
import numpy as np

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""

    total_key_length = 2**total_cube_size  #one key per cell

    key_int = int(key_hex, 16)
    
    random_key = []
    for i in range(total_key_length):
        key_shifts = []
        for dim in range(dimension):
          shift_val = (key_int >> (i*dimension+dim) * bit_shift_size) & (2**bit_shift_size -1)
          key_shifts.append(shift_val)
        random_key.append(key_shifts)

    cube_side_length = 2 ** (total_cube_size // dimension)
    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)


    def rotate_bit_shift(hypercube, random_key, shift_dimension, line_index):
        shifted_hypercube = hypercube.copy()
        
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = random_key[key_index][shift_dimension]
        
        line_slice = list(line_index)
        for cell in range(cube_side_length):
          
          temp_slice = list(line_slice)
          temp_slice[shift_dimension] = cell
          
          new_coords = list(temp_slice)
          new_coords[shift_dimension] = (new_coords[shift_dimension] + shift_amount) % cube_side_length
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    for line_index in np.ndindex(*hypercube.shape): #loops through each line first
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = random_key[key_index][shift_dimension]
        print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, random_key, shift_dimension, list(line_index))
        shift_history.append((shift_dimension, line_index, shift_amount, shifted_hypercube.copy(),key_index))

    for i in range(len(shift_history)-2, len(shift_history)):
      shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
      print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      print(shifted_hypercube)

    print("\nRandom Key:")
    print(random_key)


    return hypercube, shifted_hypercube


dimension = 5
total_cube_size = 5
bit_shift_size = 1 #2^1 shifting by 0 to 1
cube_data = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1]

key_hex = "badd0661e5" 
hypercube, shifted_hypercube = create_hypercube_and_shift(
    dimension, total_cube_size, key_hex, bit_shift_size, cube_data
)

print(f"Original Hypercube shape: {hypercube.shape}")
print(f"Shifted Hypercube shape: {shifted_hypercube.shape}")