import random
import sys
import math

dimension = 4
#params in bits
total_cube_size = 12 #2^12 = 2^3^4 (8 length 4d hypercube)
bit_shift_size = 3 #2^3 is all you need for 8 length
total_key_length = total_cube_size+bit_shift_size+math.ceil(math.log2(dimension)) #multiply by bit shift and 2^2 = 4 so enough for each dimension
#total_cube_size = 24 #2^24 = 2^8^3 (256 length cube) = 2^6^4 (64 length 4d hypercube) 

#creates enough random key parts to fill Total
random_key = [random.randint(0,1) for i in range(total_key_length)]

#example hypercube: [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]]


