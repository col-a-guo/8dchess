import random
import sys
import math
import numpy as np

def create_hypercube_and_shift(dimension, total_cube_size, num_ones, seed, bit_shift_size):
    """Creates a hypercube, populates with 1s, and applies bit shifts."""
    random.seed(seed)
    np.random.seed(seed)

    total_key_length = 2**total_cube_size  #one key per cell

    random_key = [
        [random.randint(0, 2**bit_shift_size - 1) for _ in range(dimension)] #(dimension) bit shifts per cell
        for _ in range(total_key_length)
    ]


    cube_side_length = 2 ** (total_cube_size // dimension)
    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.zeros(hypercube_shape, dtype=int)


    #populate 1s into random positions
    indices = random.sample(range(hypercube.size), min(num_ones, hypercube.size))  #ensure not more than hypercube size
    for index in indices:
        coords = np.unravel_index(index, hypercube.shape)
        hypercube[coords] = 1

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

    #print the last two shifts
    for i in range(len(shift_history)-2, len(shift_history)):
      shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
      print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      print(shifted_hypercube)

    print("\nRandom Key:")
    print(random_key)

    return hypercube, shifted_hypercube




#params in bits
dimension = 3
total_cube_size = int(2*dimension) #2^6 = 2^2^3 (4 length 3d cube)
bit_shift_size = 2 #2^2 shifting by 0 to 3


#other params
num_ones = int((2**total_cube_size)/8)
seed = 1


hypercube, shifted_hypercube = create_hypercube_and_shift(
    dimension, total_cube_size, num_ones, seed, bit_shift_size
)

print(f"Original Hypercube shape: {hypercube.shape}")
print(f"Shifted Hypercube shape: {shifted_hypercube.shape}")