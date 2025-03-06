import numpy as np
import random
succ_counter = 0
for i in range(1000):
    cube_data = bytearray(random.randint(0, 255) for _ in range(256))

    hypercube = np.array(cube_data).reshape(4,4,4,4)

    flattened = hypercube.flatten()

    byte_back = bytearray(flattened)
    if byte_back==cube_data:
        succ_counter+=1
print(succ_counter)

import numpy as np

def iterate_ndindex_backwards_generator(*shape):
    """
    Generates indices in reverse order, similar to np.ndindex, but accepts
    shape as separate arguments using the asterisk.
    """
    shape = tuple(shape)  # Convert the arguments into a tuple
    ranges = [range(s - 1, -1, -1) for s in shape]

    indices = [r.start for r in ranges]

    while True:
        yield tuple(indices)

        i = len(shape) - 1
        while i >= 0:
            if indices[i] > ranges[i].stop:
                indices[i] = ranges[i].start
                i -= 1
            else:
                break
        else:
            return

        if i >= 0:
            indices[i] += ranges[i].step
import numpy as np

def iterate_ndindex_backwards_generator(*shape):
    """
    Generates indices in reverse order, similar to np.ndindex, but accepts
    shape as separate arguments using the asterisk.
    """
    shape = tuple(shape)  # Convert the arguments into a tuple
    ranges = [range(s) for s in shape] # Regular range this time

    indices = [s - 1 for s in shape] #Start at the end of the dimension

    while True:
      yield tuple(indices)

      i = len(shape) - 1
      while i >= 0:
        indices[i] -= 1 # Subtract from the index
        if indices[i] < 0: #Check against 0
          indices[i] = shape[i] - 1 #reset to end of dimension
          i -= 1
        else:
          break
      else:
        # All dimensions have reached the end, so stop iterating
        return



# Example Usage:
shape = (2, 3, 4)

print("Forward ndindex:")
for index in np.ndindex(*shape):
    print(index)

print("\nBackward ndindex:")
for index in iterate_ndindex_backwards_generator(*shape): # Pass with asterisk
    print(index)


# Verification:
forward_indices = list(np.ndindex(*shape))
backward_indices = list(iterate_ndindex_backwards_generator(*shape))

assert forward_indices[::-1] == backward_indices, "Backward iteration is incorrect!"

print("\nAssertion Passed: Backward iteration matches the reverse of forward iteration!")