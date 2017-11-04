import nphash
import numpy as np
import datetime as dt

import nphash
print nphash.fnv1a(np.array([[1,2,3,4,5,6]]), axis=1)
print nphash.fnv1a(np.array([1,2,3,4,5,6]), axis=0)

# compare hash along different axis of an one element array
print "compare one elment array hashes"
print "  fnv1a([[10]], axis=1)"
print "   ", nphash.fnv1a(np.array([[10]]), axis=1)
print "  fnv1a([10], axis=0)"
print "   ", nphash.fnv1a(np.array([10]), axis=0)
print "  fnv1a([10])"
print "   ", nphash.fnv1a(np.array([10]))


# Test lots of data
print "do lots of data"
a = np.ones((500000, 10)).astype(np.float32)
count = reduce(lambda x,y: x*y, a.shape, 1) * 4
start = dt.datetime.now()
b = nphash.fnv1a(a)
time = dt.datetime.now() - start
print "  time", time.total_seconds()
print "  mb per second", float(count) / time.total_seconds() / 1024 / 1024

# test every axes
print "try every axis"
a = np.random.random((2,3,4,5)) * 39487921
a = a.astype(np.int32)
print "  input shape", a.shape
b = nphash.fnv1a(a, axis=3)
print "  result shape after hash along axis 3", b.shape
b = nphash.fnv1a(a, axis=2)
print "  result shape after hash along axis 2", b.shape
b = nphash.fnv1a(a, axis=1)
print "  result shape after hash along axis 1", b.shape
b = nphash.fnv1a(a, axis=0)
print "  result shape after hash along axis 0", b.shape
