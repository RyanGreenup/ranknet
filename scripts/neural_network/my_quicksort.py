# https://www.geeksforgeeks.org/python-program-for-quicksort/
DEBUG = False

def split(values, left, high): 
	# Define the leftmost value
	l = (left-1)
	# Set the right value as the pivot
	pivot = values[high]	 

	for q in range(left, high): 
		# Only move smaller values left
		if values[q] <= pivot: 
			# +1 next left element
			l = l+1
			# Swap the current element onto the left
			values[l], values[q] = values[q], values[l] 

    # Swap the pivot value into the left position from the right
	values[l+1], values[high] = values[high], values[l+1] 
	return (l+1) 


def quicksort(values, left, right): 
	if len(values) == 1: 
		return values 
	if right > left:
		# pi is the index of where the pivot was moved to
		# It's position is now correct
		pi = split(values, left, right) 

		# Do this again for the left and right parts
		quicksort(values, left, pi-1) 
		quicksort(values, pi+1, right) 


if DEBUG:
	for i in range(3):
		import random
		values = random.sample(range(9), 7)
		n = len(values) 
		print(values)
		quicksort(values, 0, n-1) 
		print("==>", values) 
