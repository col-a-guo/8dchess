import numpy as np
import random
import cProfile
import pstats
from math import pi

def generate_random_data(size):
    return bytearray(random.getrandbits(8) for _ in range(size))

def bitarray_from_bytearray(bytearr):
    return np.unpackbits(np.array(bytearr, dtype=np.uint8))

def create_hypercube_of_squares(bitarr, hypercube_length, square_length, num_dimensions):
    """Creates a hypercube where each cell contains a square (square_length x square_length)."""
    cube_size = hypercube_length ** num_dimensions * (square_length * square_length)
    reshaped = bitarr[:cube_size].reshape((hypercube_length,) * num_dimensions + (square_length, square_length))
    return reshaped

def create_index_cube(hypercube_length, num_dimensions):
    """Creates a hypercube where each cell contains its own multi-dimensional index."""
    indices = np.indices((hypercube_length,) * num_dimensions)
    index_cube = np.stack(indices, axis=-1)
    return index_cube

def apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions):
    """Applies rotations based on the key to the index cube."""
    rotated_index_cube = np.copy(index_cube)  # Avoid modifying the original
    index = 0
    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        for dim in range(num_dimensions):
            shift = key[index] % hypercube_length
            rotated_index_cube = np.roll(rotated_index_cube, shift, axis=dim)
            index += 1
            if index >= len(key):
                return rotated_index_cube
    return rotated_index_cube

def reverse_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions):
    """Applies reverse rotations based on the key to the index cube."""
    rotated_index_cube = np.copy(index_cube)  # Avoid modifying the original
    index = len(key) - 1
    for coords in reversed(list(np.ndindex((hypercube_length,) * num_dimensions))):
        for dim in reversed(range(num_dimensions)):
            shift = key[index] % hypercube_length
            rotated_index_cube = np.roll(rotated_index_cube, -shift, axis=dim)
            index -= 1
            if index < 0:
                return rotated_index_cube
    return rotated_index_cube

def encrypt_byte_array(byte_array, key, hypercube_length, square_length, num_dimensions):
    """Encrypts the byte array into a hypercube of squares using the index cube rotation."""
    bit_array = bitarray_from_bytearray(byte_array)
    cube_of_squares = create_hypercube_of_squares(bit_array, hypercube_length, square_length, num_dimensions)
    index_cube = create_index_cube(hypercube_length, num_dimensions)
    rotated_index_cube = apply_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
    encrypted_cube = np.zeros_like(cube_of_squares)  # Initialize with zeros, same shape and type

    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        original_coords = tuple(rotated_index_cube[coords])
        encrypted_cube[coords] = cube_of_squares[original_coords]

    return encrypted_cube

def decrypt_hypercube(encrypted_cube, key, hypercube_length, square_length, num_dimensions):
    """Decrypts the hypercube of squares back into a byte array using reversed index cube rotation."""
    index_cube = create_index_cube(hypercube_length, num_dimensions)
    rotated_index_cube = reverse_rotations_to_index_cube(index_cube, key, hypercube_length, num_dimensions)
    decrypted_cube = np.zeros_like(encrypted_cube)  # Initialize with zeros, same shape and type

    for coords in np.ndindex((hypercube_length,) * num_dimensions):
        original_coords = tuple(rotated_index_cube[coords])
        decrypted_cube[coords] = encrypted_cube[original_coords]  # Fill by reverse lookup

    # Flatten the cube back into a bit array
    bit_array = decrypted_cube.flatten()

    # Pack the bit array back into a byte array.  Must be multiple of 8
    bit_array = bit_array[:len(bit_array) - (len(bit_array) % 8)]
    byte_array = np.packbits(bit_array).tobytes()

    return byte_array

def pad_with_pi(data, required_size):
    """Pads the data with digits of pi until it reaches the required size."""
    pi_digits = str(pi).replace('.', '')  # Remove decimal point
    padded_data = bytearray(data)

    pi_index = 0
    while len(padded_data) < required_size:
        digit_pair = pi_digits[pi_index:pi_index + 2]
        if len(digit_pair) == 2:
            try:
                padded_data.append(int(digit_pair)) #convert pi digit pairs to bytes
            except ValueError:
                padded_data.append(0) #if there is an error, pad with a zero
        else:
            padded_data.append(0) #pad with 0 if you can't get 2 digits.

        pi_index = (pi_index + 2) % len(pi_digits) #cycle through the digits of pi
    return padded_data[:required_size]  # Truncate if necessary

# Constants
hypercube_length, square_length, num_dimensions = 8, 32, 3

# Calculate required sizes
data_size = hypercube_length**num_dimensions * square_length*square_length // 8
key_size = (hypercube_length**num_dimensions) * num_dimensions

file_path = 'ORV.txt'

# Open the file and read its contents
with open(file_path, 'rb') as file:
    myfile = file.read()

# Now file_contents holds the content of the text file
print(myfile)



# print(myfile)
# Generate random data and key
original_byte_array = bytearray(myfile) #smaller than it should be to test padding
key = generate_random_data(key_size //2) # smaller than it should be

# Pad with pi
original_byte_array = pad_with_pi(original_byte_array, data_size)
key = pad_with_pi(key, key_size)


# Run the code with cProfile
with cProfile.Profile() as pr:
    # Encrypt the data
    encrypted_cube = encrypt_byte_array(original_byte_array, key, hypercube_length, square_length, num_dimensions)

    # Decrypt the data
    decrypted_byte_array = decrypt_hypercube(encrypted_cube, key, hypercube_length, square_length, num_dimensions)

    # Verify the decryption
    if original_byte_array == decrypted_byte_array:
        print("Decryption successful!")
        
        original_file = decrypted_byte_array.decode('utf-8')
        print(original_file)
    else:
        print("Decryption failed.")

# Print cProfile results
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()