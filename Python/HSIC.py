"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf
[2]: https://link.springer.com/chapter/10.1007/11564089_7
[3]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence


"""
import matplotlib.pyplot as plt
import scipy.linalg
import numpy
import math
import sys

def centering(M):
    """
    Calculate the centering matrix, needed for
    [1] eq (9)
    """
    n = M.shape[0]
    unit = numpy.ones([n, n])
    identity = numpy.eye(n)
    H = identity - unit/n

    return numpy.matmul(M, H)

def gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    |x_i - x_j|**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = numpy.matmul(x, x.T)
    xnorm = numpy.diag(xxT) - xxT + (numpy.diag(xxT) - xxT).T
    if sigma is None:
        #mdist = numpy.median(xnorm[xnorm != 0]) #Not even faster. why do it like this?
        mdist = numpy.median(xnorm)
        sigma = math.sqrt(mdist*0.5)

    # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    numpy.exp(KX, KX)
    return KX


def dHSIC_calc0(K_list):
    """
    Doesn't work
    """
    n_k = len(K_list)

    length = K_list[0].shape[0]
    dterm1 = numpy.ones((length, length))
    dterm2 = numpy.ones((length, length))
    dterm3 = numpy.ones((length, length))

    # Works:
    k = K_list[0]
    l = K_list[1]
    dterm1 = numpy.sum(numpy.multiply(k, l))
    print(dterm1)
    dterm2 = 1/(length**4) * numpy.sum(k) * numpy.sum(l)
    print(dterm2)
    dterm3 = 2/(length**3) * numpy.sum(numpy.multiply(k, l))
    print(dterm3)
    dterm3 = 2/(length**3) * numpy.sum(numpy.matmul(k, l))
    print(dterm3)
    for j in range(0, n_k):
        K_j = K_list[j]
        dterm1 = numpy.sum(numpy.multiply(dterm1, K_j))
        dterm2 = 1/(length**4) * numpy.sum(dterm2) * numpy.sum(K_j)
        dterm3 = 2/(length**3) * numpy.sum(numpy.multiply(dterm3, K_j))
    print("suggested term 1")
    print(dterm1)
    print("suggested term 2")
    print(dterm2)
    print("suggested term 3")
    print(dterm3)
    return (1/(length**2)*dterm1 + dterm2 - dterm3)


def dHSIC_calc(K_list):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [3] Definition 2.6
    """
    if not isinstance(K_list, list):
        K_list = list(K_list)

    n_k = len(K_list)

    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0/length

    for j in range(0, n_k):
        K_j = K_list[j]
        term1 = numpy.multiply(term1, K_j)
        term2 = 1.0/length/length*term2*numpy.sum(K_j)
        term3 = 1.0/length*term3*K_j.sum(axis=0)

    term1 = numpy.sum(term1)
    term3 = numpy.sum(term3)
    #print("correct term 1")
    #print(term1)
    #print("correct term 2")
    #print(term2)
    #print("correct term 3")
    #print(term3)
    dHSIC = (1.0/length)**2*term1+term2-term3
    return dHSIC

def HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return numpy.trace(numpy.matmul(centering(gaussian_grammat(x)),centering(gaussian_grammat(y))))/n/n

def dHSIC(*argv):
    assert len(argv) > 1, "dHSIC requires at least two arguments"

    if len(argv) == 2:
        x, y = argv
        return HSIC(x, y)

    #if len(argv) > 2
    K_list = [gaussian_grammat(_arg) for _arg in argv]
    return dHSIC_calc(K_list)


if __name__ == "__main__":

    # --- Data
    D = 5
    N = 100

    #X = numpy.array([numpy.linspace(-1, 1, N) for _ in range(D)]).T
    X = numpy.array([numpy.random.uniform(-1, 1, N) for _ in range(D)]).T
    TWO_D = 2*numpy.array(range(D))
    Y = numpy.matmul(numpy.multiply(X, X), TWO_D)
    # ---

    # --- Test dHSIC calculations
    K_list = [gaussian_grammat(_x) for _x in [X, Y]]
    print(dHSIC_calc(K_list))
    #print(dHSIC_calc0(K_list))
    print(dHSIC(X, Y))

    #print(dHSIC(X, Y, X, Y))
    #print(dHSIC(X))
