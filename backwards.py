
import sys
import random
import math
import numpy as np

def iterate_ndindex_backwards_generator(shape):
    """Iterates through an np.ndindex object backwards using a generator (memory-efficient).

    Args:
        shape: The shape of the ndarray to index. Can be a tuple or a list.

    Yields:
        Tuples representing the indices, in reverse order of np.ndindex.
    """
    shape = tuple(shape)  # Ensure it's a tuple
    total_size = np.prod(shape, dtype=np.intp) # Calculate total number of indices

    for i in range(total_size - 1, -1, -1): # Iterate backwards through the linear index
        index = np.unravel_index(i, shape)  # Convert linear index to multi-dimensional index
        yield index


def hex_to_bits(hex_string):
    # Initialising hex string 
    
    # Printing initial string 
    print ("Initial string", hex_string) 
    
    # Code to convert hex to binary 
    n = int(hex_string, 16) 
    bStr = '' 
    while n > 0: 
        #calculate the remainder when divided by 2; last bit. Then append
        bStr = str(n % 2) + bStr 
        #move to the next bit
        n = n >> 1   
    res = bStr 
    return(res)


def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""

    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str_to_flip = hex_to_bits(key_hex)


    # take each set of bits associated with a shift, and get the "negative" of it, e.g. if cube length is 8, then shifting 5 then shifting 3 is doing nothing, i.e. 5 = -3
    key_str = [0 for i in range(2**total_cube_size*dimension)]
    for shift_index in range(2**total_cube_size*dimension):
        bits_to_check = key_str_to_flip[shift_index*bit_shift_size:shift_index*bit_shift_size+bit_shift_size]
        new_int_shift = cube_side_length-int(bits_to_check, 2)
        new_bit_shift = bin(new_int_shift)[2:]
        key_str[shift_index] = new_bit_shift

    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*2**bit_shift_size #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments
            key_shifts.append(key_str[cube_offset+dimension_offset:cube_offset+dimension_offset+2**bit_shift_size]) #+2**bit_shift_size gives the bits needed to shift
        arbitrary_key.append(key_shifts)

    hypercube_shape = (cube_side_length,) * dimension
    hypercube = np.array(cube_data).reshape(hypercube_shape)


    def rotate_bit_shift(hypercube, arbitrary_key, shift_dimension, line_index):
        """hypercube: original hypercube
        arbitrary_key: key we're rotating with
        shift_dimension: e.g. x y z
        line_index: grabbed via np.ndindex(hypercube.shape); tells us the line in the shift dimension's coordinates
        e.g. if shift dim = 1, would look like [0,0,0], [0,1,0], [0,2,0]..."""
        shifted_hypercube = hypercube.copy() #make a copy so we can do stuff with it
        
        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape) #ravel = un un ravel, i.e. wrap up a line into a cube
        shift_amount = arbitrary_key[key_index][shift_dimension] #find the shift amount (see above)
        
        line_slice = list(line_index) #find the initial coordinate for the line we're slicing
        for cell in range(cube_side_length):
          
          temp_slice = list(line_slice) #make a copy of the slice so it doesn't self reference
          temp_slice[shift_dimension] = cell 
          
          new_coords = list(temp_slice)
          #for demonstration, print before and after some specific times
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("Before:")
            print(new_coords)


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount)) % cube_side_length
          

          #for demonstration
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("After:")
            print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    for line_index in iterate_ndindex_backwards_generator(hypercube.shape): #loops through each line backwards
      print(line_index)
      for shift_dimension in range(dimension,0,-1): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration``
        if ((shift_amount == "0001") or (shift_amount == "0101")):
          print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))
        shift_history.append((shift_dimension, line_index, shift_amount, shifted_hypercube.copy(),key_index))
    #prints last two
    for i in range(len(shift_history)-2, len(shift_history)):
      shift_dimension, line_index, shift_amount, shifted_hypercube, key_index = shift_history[i]
      print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      print(shifted_hypercube)

    # print("\nRandom Ahh Key:")
    # print(arbitrary_key)


    return hypercube, shifted_hypercube
