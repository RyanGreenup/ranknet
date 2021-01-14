import numpy

def misclassification_rate(target: numpy.ndarray, output: numpy.ndarray):
    """
    docstring
    """
    if type(target) is not numpy.ndarray or type(output) is not numpy.ndarray:
        print("Both input values must be numpy arrays, Exiting")
        raise ValueError("Require Numpy Arrays")

    try: 
        target = target.reshape(1, -1).astype(int)
        output = output.reshape(1, -1).astype(int)
    except ValueError:
        print("Input values must both be Numpy Arrays filled with integers!")

    if numpy.shape(target) == numpy.shape(output):
        misclassification_rate = numpy.average(target == output)
    else:
        raise IndexError('Target and Output Arrays must be the same shape')

    return misclassification_rate




