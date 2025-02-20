import sys
import random
import math
import numpy as np

def hex_to_bits(hex_string):
    # Initialising hex string 
    
    # Printing initial string 
    print ("Initial string", hex_string) 
    
    # Code to convert hex to binary 
    n = int(hex_string, 16) 
    bStr = bin(n)[2:] #changed here
    #res = bStr
    return(bStr)
    
def pad_right(text, length, char=' '):
    """Pads the given string on the right with the specified character until it reaches the desired length.

    Args:
        text: The string to pad.
        length: The desired total length of the padded string.
        char: The character to use for padding (default is space).

    Returns:
        The padded string, or the original string if it's already long enough.
    """
    if len(text) >= length:
        return text
    return text + char * (length - len(text))

    

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""

    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size)*4/4)
      print(size)
      if len(key_hex) < size:
        key_temp = pad_right(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)
    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str = pad_right(key_str, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")

    print(key_str)
    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size) #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments
            key_shifts.append(key_str[cube_offset+dimension_offset:cube_offset+dimension_offset+(2**bit_shift_size)]) #+2**bit_shift_size gives the bits needed to shift
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


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

          #for demonstration
          if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
            print("After:")
            print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    
    shift_history = []

    for line_index in np.ndindex(*hypercube.shape): #loops through each line first
      print(line_index)
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration
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

def reverse_hypercube_and_reverse_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""
    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size)*4/4)
      if len(key_hex) < size:
        key_temp = pad_right(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)

    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str_to_flip = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str_to_flip = pad_right(key_str_to_flip, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")



    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size) #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments

            bits_to_check = key_str_to_flip[cube_offset+dimension_offset:cube_offset+dimension_offset+bit_shift_size]
            
            new_int_shift = (cube_side_length-int(bits_to_check, 2)) % cube_side_length
            new_bit_shift = bin(new_int_shift)[2:].zfill(bit_shift_size)
            print(new_bit_shift)
            key_shifts.append(new_bit_shift) #+2**bit_shift_size gives the bits needed to shift
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


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

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
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

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

#power of 2 params
dimension = 3
total_cube_size = 3 #2^3
bit_shift_size = 2 #2^2 shifting by 0 to 3
cube_data = [0,1,1,0,1,1,0,1] #total 2^3 = 8 entries 

key_hex = ""
for i in range(12):
   key_hex+=str(random.randint(0,9))
hypercube, shifted_hypercube = create_hypercube_and_shift(
    dimension, total_cube_size, key_hex, bit_shift_size, cube_data
)

print(hypercube)

hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
    dimension, total_cube_size, key_hex, bit_shift_size, shifted_hypercube
)

print(og_hypercube)