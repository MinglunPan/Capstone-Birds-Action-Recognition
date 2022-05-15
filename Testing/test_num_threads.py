import numpy
import sys
import timeit

try:
    import numpy.core._dotblas
    print('FAST BLAS')
except ImportError:
    print('slow blas')

print("version:", numpy.__version__)
print()

x = numpy.random.random((1000,1000))

setup = "import numpy; x = numpy.random.random((1000,1000))"
count = 5

t = timeit.timeit("numpy.dot(x, x.T)", setup=setup,number=100)
print("dot:", t/100, "sec")
