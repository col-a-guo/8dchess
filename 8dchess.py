import random
import sys
import math
import numpy as np

def create_hypercube_and_shift(dimension, total_cube_size, num_ones, seed, bit_shift_size):
    """Creates a hypercube and a random key, populates with random 1s, and applies bit shifts. In final algorithm, will be modified to take specific hypercube and key; 8dchessexample.py starts this process."""
    random.seed(seed) #sets random seed for replication
    np.random.seed(seed) #sets seed for numpy

    total_key_length = 2**total_cube_size  #sets initial key size to total size of the cube; "total_cube_size" is in bits

    #Create (dimension) bits per cell, to shift in a given dimension. Max shift = 2**bit_shift_size, min shift = 0 
    random_key = [
        [random.randint(0, 2**bit_shift_size - 1) for _ in range(dimension)] #(dimension) bit shifts per cell
        for _ in range(total_key_length)
    ]

    # Determines the size the cube should be
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    # Sets the cube shape, e.g. 4x4x4x4x4
    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.zeros(hypercube_shape, dtype=int)

    #populate 1s into random positions
    indices = random.sample(range(hypercube.size), min(num_ones, hypercube.size))  #ensure not more than hypercube size

    #unwraps cube e.g. 4x4x4x4x4 into 1024, so that 1s can be populated
    for index in indices:
        coords = np.unravel_index(index, hypercube.shape)
        hypercube[coords] = 1
        
    ### Here's the big one for dimension: ###
    ### How to bit shift based on a given dimension ###
    def rotate_bit_shift(hypercube, random_key, shift_dimension, line_index):
        #start by creating a copy so we can reference bits
        shifted_hypercube = hypercube.copy()

        #find the correct cell by un-unraveling
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)

        #find the correct amount to shift by
        ### The only thing we have to change from same key per dimension, is adding [shift_dimension] to this line (and the key, and followups): ###
        shift_amount = random_key[key_index][shift_dimension]

        #slice exactly the line of bits we want to shift (e.g. [0,0,0,0,0],[1,0,0,0,0],[2,0,0,0,0],[3,0,0,0,0])
        line_slice = list(line_index)
        for cell in range(cube_side_length):

            #figure out what the old coordinates had in them
            temp_slice = list(line_slice)
            temp_slice[shift_dimension] = cell
            
            new_coords = list(temp_slice)

            #put each old coordinate in the cell (shift_amount) in front of it, with wrapping, rotating by (shift_amount)
            new_coords[shift_dimension] = (new_coords[shift_dimension] + shift_amount) % cube_side_length
            shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube

    #Copying and history indexing so that I can print them for debugging; doesn't interact with return
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    ### Now we just go through each cell, shifting in each dimension per cell ###
    for line_index in np.ndindex(*hypercube.shape): #loops through each line first
        for shift_dimension in range(dimension): #then shifts all dimensions in each line.
            
            key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
            
            #more printing; doesn't interact with return
            shift_amount = random_key[key_index][shift_dimension]
            print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")

            ### Apply rotate_bit_shift ###
            shifted_hypercube = rotate_bit_shift(shifted_hypercube, random_key, shift_dimension, list(line_index))

            #history saving; doesn't interact with return
            shift_history.append((shift_dimension, line_index, shift_amount, shifted_hypercube.copy(),key_index))

    #print the last two shifts
    for i in range(len(shift_history)-2, len(shift_history)):
        shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
        print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        print(shifted_hypercube)

    #and the random key
    print("\nRandom Key:")
    print(random_key)

    return hypercube, shifted_hypercube




#parameters in bits
dimension = 5
total_cube_size = int(2*dimension) #2^10 = 2^2^5 (4 length 3d cube)
bit_shift_size = 2 #2^2 shifting by 0 to 3


#other params
num_ones = int((2**total_cube_size)/8)
seed = 1


hypercube, shifted_hypercube = create_hypercube_and_shift(
    dimension, total_cube_size, num_ones, seed, bit_shift_size
)

print(f"Original Hypercube shape: {hypercube.shape}")
print(f"Shifted Hypercube shape: {shifted_hypercube.shape}")
