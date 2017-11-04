nphash
======

Compute a hash on your numpy array data.

Either compute per element hashes or specify an axis to compute the hash along. 
This works similar to the syntax of numpy.sum and others.


Build 'n install
----------------

A simple 
```
python setup.py install
```
should do the trick.



Usage
-----
```
import nphash
import numpy as np

data = np.random.random((10,3))

# element wise hash
print nphash.fnv1a(data)

# along axis 1
print nphash.fnv1a(data, axis=1)
```




