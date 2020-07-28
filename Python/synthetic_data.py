import sys
import numpy
import scipy
import warnings
import random


def make_data_seq(d, n, sigma=0.01):

    x = numpy.reshape(numpy.linspace(-1, 1, d*n), (d, n)).T
    noise = numpy.random.normal(0, sigma, (n,d))
    x = x + noise
    y = numpy.matmul(numpy.multiply(x, x), numpy.ones(d))


    return x, y


def make_data_random(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    two_d = 2*numpy.array(range(d))

    #x = numpy.array([numpy.linspace(-1, 1, n) for _ in range(d)]).T
    x = numpy.array([numpy.random.uniform(-1, 1, n) for _ in range(d)]).T
    epsilon = numpy.random.uniform(-0.2, 0.2, n)
    y = numpy.matmul(numpy.multiply(x, x), two_d) #+ epsilon

    return x, y

def make_data_noisy(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    if d !=3:
        print("Please use only d=3 for noisy data function")
        sys.exit()

    x1 = numpy.random.normal(0, 1, n)
    x2 = numpy.random.normal(0, 3, n)
    x3 = numpy.random.normal(0, 4, n)
    y = x1 + numpy.random.normal(0, 2, n)

    x = numpy.vstack((x1, x2, x3)).T

    return x, y

def make_data_step(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    two_d = 2*numpy.array(range(d))

    x = numpy.array([numpy.random.uniform(-1, 1, n) for _ in range(d)]).T
    y = numpy.matmul([[(-0.5 < _xi and _xi < 0.5) for _xi in _x] for _x in x], two_d)

    return x, y

def make_data_xor_discrete(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    if d !=2:
        print("Please use only d=2 for XOR function")
        sys.exit()
    # no d currently implemented. d=2.

    x1 = numpy.random.uniform(-1, 1, n)
    x1 = [0 if _x<0 else 1 for _x in x1]
    x2 = numpy.random.uniform(-1, 1, n)
    x2 = [0 if _x<0 else 1 for _x in x2]

    y = numpy.logical_xor(x1, x2)
    y = numpy.array([1 if _y else 0 for _y in y])

    x = numpy.vstack((x1, x2)).T

    return x, y

def make_data_xor_discrete_discrete(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    if d !=2:
        print("Please use only d=2 for XOR function")
        sys.exit()
    # no d currently implemented. d=2.

    test = numpy.repeat([0,1],int(n/2))
    x1 = random.sample(list(test),n)
    x2 = random.sample(list(test),n)
    y = numpy.array([1 if _a else 0 for _a in numpy.logical_xor(x1,x2)])

    x = numpy.vstack((x1, x2)).T

    return x, y

def make_data_xor(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    if d !=2:
        print("Please use only d=2 for XOR function")
        sys.exit()
    # no d currently implemented. d=2.

    x1 = numpy.random.uniform(-1, 1, n)
    x2 = numpy.random.uniform(-1, 1, n)
    x = numpy.vstack((x1, x2)).T

    y = numpy.array(
            [_x1*(_x1 > 0 and _x2 < 0) + _x2*(_x1 < 0 and _x2 > 0)
           + _x1*(_x1 < 0 and _x2 < 0) - _x2*(_x1 > 0 and _x2 > 0)
            for _x1, _x2 in zip(x1, x2)])

    return x, y

def make_data_harmonic(d, n):
    if d > n:
        warnings.warn("Warning: More features than samples!", UserWarning)

    two_d = 2*numpy.array(range(d))

    x = numpy.array([numpy.random.uniform(-numpy.pi, numpy.pi, n) for _ in range(d)]).T
    y = numpy.matmul(scipy.cos(x), two_d)

    return x, y

def make_data(d, n, data_type):
    if data_type is "step":
        return make_data_step(d, n)

    elif data_type is "harmonic":
        return make_data_harmonic(d, n)
    elif data_type is "random":
        return make_data_random(d, n)
    elif data_type is "xor":
        return make_data_xor(d, n)

    elif data_type is "xor_discrete":
        return make_data_xor_discrete(d, n)

    elif data_type is "xor_discrete_discrete":
        return make_data_xor_discrete_discrete(d, n)

    print(f"Data type {data_type} is not implemented")
    sys.exit()
