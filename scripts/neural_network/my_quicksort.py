import random

values = [2, 1, 7, 8, 5, 0, 4, 6, 3]
# values = random.sample(range(9), 9)
N = len(values)



def partition(values, left, right):
    #region
    '''
    Take a list of values and rearrange so that the first term is in the centre, terms less than the first entry
    appear to the left and all terms greater appear on the right, e.g.

        [5, 3, 1, 6, 8, 9] => 5: [3, 1, 5, 6, 8, 9]
    
    The order of the terms around the "pivot" don't matter, so the following would also be acceptable:

        [5, 3, 1, 6, 8, 9] => 5: [1, 3, 5, 8, 9, 6]

    Parameters:
        values (list): A list of values that need to be sorted
    Output: 

        pi (int): An integer describing the index value of the pivot point,
                  i.e. the location that the first element was moved to.
    '''
    #endregion
    l = left      # Current left Most Value
    p = values[0] # Use the first value as a pivot point

    for i in range(left, right):  # TODO and i not 0 # Skip the first value
        if values[i] <= p:
            print("Swapping index ", i, " and ", l)
            swap(values, i, l)   # If the value is less than
            l+=1                    # the pivot, swap it with a left term
    
    swap(values, 0, l-1) # Swap the pivot out of the 1st index into correct spot
    return l    # Return the index of the pivot which should be straight after the left most value

def swap(values, i, j):
    ival = values[i]
    jval = values[j]
    values[i] = jval
    values[j] = ival



print(values)
pi = partition(values, 0, N)
print(pi)
print(values)



