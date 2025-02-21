import sys
import random
import math
import numpy as np

def hex_to_bits(hex_string):
    # Initialising hex string 
    
    # Printing initial string 
    # print ("Initial string", hex_string) 
    
    # Code to convert hex to binary 
    # n = int(hex_string, 16)
    # bStr = bin(n)[2:] #changed here

    #if hex_string is already a string of bits (which it is now), then don't change it
    #res = bStr
    return("".join(hex_string))
    
def pad_right_list(lst, length, value=None):
    """Pads the given list on the right with the specified value until it reaches the desired length.

    Args:
        lst: The list to pad.
        length: The desired total length of the padded list.
        value: The value to use for padding (default is None).

    Returns:
        The padded list, or the original list if it's already long enough.
    """
    if len(lst) >= length:
        return lst
    return lst + [value] * (length - len(lst))

    

def create_hypercube_and_shift(dimension, total_cube_size, key_hex, bit_shift_size, cube_data):
    """Creates a hypercube, populates with specific data, and applies bit shifts using a specific key."""

    def pad_or_truncate_key(key_hex):
      key_temp = key_hex
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size))
      # print(size)
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)
    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str = pad_right_list(key_str, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")

    # print(key_str)
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
          # if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
          #   print("Before:")
          #   print(new_coords)


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

          #for demonstration
          # if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
          #   print("After:")
          #   print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    

    for line_index in np.ndindex(*hypercube.shape): #loops through each line first
      # print(line_index)
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration
        # if ((shift_amount == "0001") or (shift_amount == "0101")):
        #   print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))

      # print(f"Shift: Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
      # print(shifted_hypercube)

    # print("\nRandom Ahh Key:")
    # print(arbitrary_key)


    return hypercube, shifted_hypercube

#############################################
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
      size = int((2**(total_cube_size))*dimension*(2**bit_shift_size))
      if len(key_hex) < size:
        key_temp = pad_right_list(key_hex, size, "0")
      
      elif len(key_hex) > size:
        key_temp = key_hex[0:size]
      
      else:
        key_temp = key_hex
      return(key_temp)

    # total_key_length = (2**(total_cube_size))*3*(2**bit_shift_size) one mini key per cell, three dimensions per mini key, then bit shift size per dimension
    #convert str to hex (base 16 int)
    cube_side_length = 2 ** (total_cube_size // dimension)
    
    key_str_to_flip = hex_to_bits(pad_or_truncate_key(key_hex))
    key_str_to_flip = pad_right_list(key_str_to_flip, int((2**(total_cube_size))*dimension*(2**bit_shift_size)), "0")



    arbitrary_key = []
    #TODO: see if ravel just works here xd
    for cube_coordinate in range(2**total_cube_size): #loop through each cell of the cube
        key_shifts = []
        for dim in range(dimension): #then each dimension e.g. x, y, z
            cube_offset = cube_coordinate*dimension*(2**bit_shift_size) #increases when cube_coordinate increments
            dimension_offset = dim*(2**bit_shift_size) #'                  ' when dim increments

            bits_to_check = key_str_to_flip[cube_offset+dimension_offset:cube_offset+dimension_offset+(2**bit_shift_size)]
            
            new_int_shift = (cube_side_length-int(bits_to_check, 2)) % cube_side_length
            new_bit_shift = bin(new_int_shift)[2:].zfill(bit_shift_size)
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
          # if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
          #   print("Before:")
          #   print(new_coords)


          new_coords[shift_dimension] = (new_coords[shift_dimension] + int(shift_amount,2)) % cube_side_length
          

          #for demonstration
          # if ((shift_amount == "0001") or (shift_amount == "0101")) and line_slice[0]==5:
          #   # print("After:")
          #   # print(new_coords)
          shifted_hypercube[tuple(new_coords)] = hypercube[tuple(temp_slice)]
          
        return shifted_hypercube
    
    shifted_hypercube = hypercube.copy()
    

    for line_index in iterate_ndindex_backwards_generator(hypercube.shape): #loops through each line backwards
      # print(line_index)
      for shift_dimension in range(dimension): #then shifts all dimensions in each line.

        key_index = np.ravel_multi_index(tuple(line_index), hypercube.shape)
        shift_amount = arbitrary_key[key_index][shift_dimension]
        #also for demonstration``
        # if ((shift_amount == "0001") or (shift_amount == "0101")):
        #   print(f"Bit shift (line {line_index}, dimension {shift_dimension}): {shift_amount}")
        
        shifted_hypercube = rotate_bit_shift(shifted_hypercube, arbitrary_key, shift_dimension, list(line_index))


    # print("\nRandom Ahh Key:")
    # print(arbitrary_key)


    return hypercube, shifted_hypercube

def bytearray_to_bits_array(byte_data):
    return [int(bit) for byte in byte_data for bit in format(byte, '08b')]

#power of 2 params
byte_data = bytearray(b'\xee\x1c\xc2\xc1\x14\x81\x10\xbc\xe9\x94\x7f\xd8\xa4\x08\x87\x84\x02\x04\x96s2.\x0bhq\xce\xc8\xae\xbd\x9e\xcc\xed\x00\x18\x06\xc5\x01i\xe4:hj\x8d\xae\x94$}\x1al9\xb4\xe1c\x05\x90\x87p\xcd\xed\x98\xe3\x8f\x1d\x80\x10\xa0\xe0g\xca\xad\x98w~.\xb9\x91]\x82\x8c\xe7\x84T\xe5\xe6^e\xbe\xe9g)\xeb:\xa9\x9f\xe6\x99\xa8\xe9\xba\xef\xd9-\xbc\x13\xd2\xa96\xccq\xc6:\xe94\xb9cz\xa9M\xe8O$\xbfP3\x96i\xcf=\xfb\xd5\x18<\xf7\xcb;w\xfa%\xd9\x08\xdc\xae\xa4a\xd2\xce\x912\xdb\xb7\xd3\xf1q\xcbC#\xdeQ29\xac\xdf\xabn\x99\xe2\xe2\xad\xc1H?@\xca[\xda?\x9b\x07\xb7n\xdf\x8c`\x96\x1f.\x07{\xdf\xa5\xce\xf4\xa9\xf9\xb2\x91/9\x1a\x96#O\x85\xe3\xb0X@\x89\xbe\x04\xc3Mc\xaa#\xe0\xaa\xea\r.\x9c\x8c[+\xac\xfcv\x07\xcd\x0c\xafs\xef\x1dr\xb2:\x99\x15\xd8)\xc7\x1cp%\x8b\x8c\x1e#B\xf5(t\xfa\x17\n\xec\x14\xe3\x8e8\x12\xc5\xc6\x0e\xb5\xdc_\x95\x908\xd5\x87\xf4\xbc\xb7\xe6@\x13\xd7\xab\x0fw\xeeO\xe7\x80\xc0\x01-\xb3\x1cpS\xc7\xf4\x81\xda\xffK\xba\x83\xcaj\x0e\xd8z?h_\xab\x0b\xfe3\xda\xfa\xb4\xeb\r\xaf\xb9(\t\xc9\xe6\xf8\xc6\xf9~r_;\x9d,r\r\x7f\xa6\xce]-\xc9*+\x9e\xfc\xba[\xdd\xb0\xb6\xefu3;\x99\xad{\xa5\x9dg\'\x9c\xb4\x7f\xb7\xf9\xae=\xf9}\xfad\xc0a\xa6\xb3\xff\xae\xf5F\x1a\x7f\xb63\xeb\xe9ri\teE\xc6E\xf0x\xff\xdb\xf6\xd22\x19\x87\xaa\x1f\xd8;q4\xb7\xbbam\xdf"\x19\xa1[cR|\x9f\xf9Hfe\x9b\x1f\x18\x10\nA\xac7\xab\x93\xddP\x18/l\xc6:Y\xa4\xdd\xc8\xe0v\x93g\x94R\xdeq+\xd2\xe9\xd4L\xcdP\xdd\x8e\xf3\\I\xf9\xab\x0f\xe9yo\x80\x80?\xe4\xd4\xd1\xefsX\x85\xf7\tG\x8c\x0c\xd546v\xc2f\'\xd5M\xff\x85k\xb1\xdej\x7f\xe5\xf3\x1c\xfc\x14{A!\xd7\xf4\xffi\x94t\x89\x96\xdd\x99-\xe5\x18&f\xf8\xc1\xed\xf8\r4\n&c/;\xc7`|\xbc\xa4\x1a\xb5\xf6+\x93H\'\xcd\xefo.rg\xa8\x1c\x0c\xc5\xc1\xed\xdb\xb6\x19\xd2KR}N\xbdR;\xfc\x0fv\x19q\xd4\xeb\xd7\x85\x88\xce^\xaa\x7f\xcb\xe5g\xeb\x7f\xd6\xf4\xcb\xfa[\xbek\x08\xbfk\xd3\xd7\xd7\xac\x043&\xfa\xef\xa6\xba\xd2\x19f\xfbvg\xbd\xe9\xa1\xaa\xd5\x88\xf4\xa2\x98\xb7s\xef+|\xd4\x14L\xf2]t[\x92\xff\xfe\xd4~\xb1\xf1Y\xc7\xcb:;\xab3s\xbf\x9a\x8d\xae\x96\x84\xfdY\xd5s:^\xf6\xff\x1a\x8c\xf7\xe5\x86_\x91rTP<Q\xdd\xdc\xf1\x02&\xf8\x13\r5\x08\xe0.K\xfd,gsx\xec\x16\x10"o\x810\xd3P\x97\x9eI\xeb\'~\x14J_%\xeb\xe4l\xdc\x94\xfd\'*\xeeT_\xf0\x1b\x14\x08\x06\x06\x0b\xf8\x02\x85\xb1\x1dk\x9f&&W\xd9\x8f\xb55\x16\xc2t\xf5\x85\x1b\xa5\xf5\xdc\x1b\xd7I\xc2\xf1\xd8\x0c D\xdf\x02a\xa6\xa1\xbb\x1d\xe6\xb99\xaf\xeda2\x16\x96@yF\x7f\xde\xae\xcf3\xe95S6=\x9di:\x1a\xdd\x1a\xea\xf4\xe4\xda4&\xed|\xfe1\xb66\x13\xa7\xac(\xdd/\xae\xf5h},\xf4\x0b\xa6\xfc)\xecW}!\x1b\xfd^\xba~\x95\xdf\xfaEl\xe5\xca\xc8g\x98\xef\xaf\xa5\n\xfbVh\xf3w\x08\xa6H#\xf7\xea\xf5\x05\xa2h{/\xb6\xed\xf7w$\t\x9b\xd7\xaaR\xe5\xf5\xfd\xde\x18\x1d\xe4\xf2\xe2\xe2\xad\xfbn\xe3b}\x8b\xa3&\xd9\x8e8\xf1\xd8\x01\x88%M{u\x8d\x0cys-\xcc=\xffO"e\xb7fKyF\x01p\xae\xc1N8\xe3\x81,\\`i\xc3\xeb\xe00F%\x8a\x1a\xacG\xaeXW\xb8\xac\xb2\xad{\xc1\xd2\xa3\'\xb7\xb9V[1\xcf\xfc|#\xecy\xb8\xe0\x08\x06')
dimension = 3
total_cube_size = 15#2^3
bit_shift_size = 2 #2^2 shifting by 0 to 3
cube_data = bytearray_to_bits_array(byte_data) #total 2^3 = 8 entries 

# print(cube_data)
if len(cube_data) < 2**(total_cube_size):
    cube_data.extend([0] * (2**(total_cube_size) - len(cube_data)))
    # key_hex = ""



key_byte = ["".join([str(random.randint(0,1)) for i in range(1)]) for i in range((2**(total_cube_size))*dimension*(2**bit_shift_size))]


hypercube, shifted_hypercube = create_hypercube_and_shift(
    dimension, total_cube_size, key_byte, bit_shift_size, cube_data
)

print("a hypercube")
print(hypercube)
hypercube, og_hypercube = reverse_hypercube_and_reverse_shift(
    dimension, total_cube_size, key_byte, bit_shift_size, shifted_hypercube
)

print("hopefully the same hypercube:\n\n")
print(og_hypercube)