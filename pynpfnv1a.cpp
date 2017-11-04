#include "Python.h"
#include "math.h"
#include <stdio.h>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include "hash_fnv1a.h"

/**
 * Perform element wise hash of all elements in in
 **/
template<typename T>
void hash_elements(PyArrayObject *in, PyArrayObject *out) {
    int outdim = PyArray_NDIM(out);

    npy_intp *outshape = PyArray_SHAPE(out);

    npy_intp *outpos = new npy_intp[outdim];
    bool done = false;
    for (int i = 0; i < outdim; ++i) {
        outpos[i] = 0;
    }

    while (!done) {
        T &val = *static_cast<T *>(PyArray_GetPtr(in, outpos));
        unsigned long hash = fnv1a(val);
        *static_cast<unsigned long *>(PyArray_GetPtr(out, outpos)) = hash;
        
        int idx = outdim - 1;
        outpos[idx] += 1;    
        while (outpos[idx] >= outshape[idx]) {
            outpos[idx] = 0;
            --idx;
            if (idx < 0) {
                done = true;
                break;
            }
            outpos[idx] += 1;
        }
    }
}

template<typename T>
void hash_axis(PyArrayObject *in, PyArrayObject *out, int axis) {
    int dim = PyArray_NDIM(in);
    int outdim = PyArray_NDIM(out);

    npy_intp *shape = PyArray_SHAPE(in);
    npy_intp *outshape = PyArray_SHAPE(out);

    npy_intp *pos = new npy_intp[dim];
    npy_intp *outpos = new npy_intp[outdim];
    bool done = false;
    for (int i = 0; i < outdim; ++i) {
        outpos[i] = 0;
    }

    while (!done) {
        for (int i = 0; i < outdim; ++i) {
            if (i < axis) {
                pos[i] = outpos[i];
            } else if (i >= axis) {
                pos[i + 1] = outpos[i];
            }
        }

        unsigned long hash = 0;
        for (int x = 0; x < shape[axis]; ++x) {
            pos[axis] = x;
            T &val = *static_cast<T *>(PyArray_GetPtr(in, pos));
            if (x == 0) {
                hash = fnv1a(val);
            } else {
                hash = fnv1a(val, hash);
            }
        }
        unsigned long &res = *static_cast<unsigned long *>(PyArray_GetPtr(out, outpos));
        res = hash;
        
        int idx = outdim - 1;
        outpos[idx] += 1;    
        while (outpos[idx] >= outshape[idx]) {
            outpos[idx] = 0;
            --idx;
            if (idx < 0) {
                done = true;
                break;
            }
            outpos[idx] += 1;
        }
    }

}



static PyObject *pyfnv1a(PyObject *self, PyObject *args, PyObject *kwrds) {

    static char *kwlist[] = {"", "axis", NULL};
    PyObject *input;
    int axis = -1;
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|i", kwlist, &input, &axis)) {
        Py_RETURN_NONE;
    }

    bool axis_defined = false;
    if (kwrds != NULL) {
        if (PyDict_Check(kwrds)) {
            PyObject *key = PyString_FromString("axis");
            if (PyDict_Contains(kwrds, key) == 1) {
                axis_defined = true;
            }
            Py_DECREF(key);
        }
    }

    PyArrayObject *inarray = reinterpret_cast<PyArrayObject *>(
            PyArray_FROM_OTF(input, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY));

    size_t nd = PyArray_NDIM(inarray);
    npy_intp *inshape = PyArray_SHAPE(inarray);
    PyArray_Descr *dtype = PyArray_DTYPE(inarray);

    size_t resultdims = nd;
    if (axis_defined) {
        resultdims = std::max(size_t(1), nd - 1);
    }

    npy_intp *resultshape = new npy_intp[nd];
    PyArrayObject *result;

    if (axis_defined) {
        // Make sure the axis exists
        if (axis < 0 || axis >= nd) {
            PyErr_SetString(PyExc_TypeError, "axis does not exist, value invalid.");
            Py_DECREF(inarray);
            Py_RETURN_NONE;
        }

        // Compute resultshape when we have an axis defined
        if (nd > 1) {
            for (size_t i = 0; i < nd; ++i) {
                if (i < axis) {
                    resultshape[i] = inshape[i];
                } else if (i > axis) {
                    resultshape[i - 1] = inshape[i];
                }
            }
        } else {
            resultshape[0] = 1;
        }
    
        result = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(resultdims, resultshape, NPY_UINT64));

        if (dtype->type == 'f') {
            // float32 array
            hash_axis<float>(inarray, result, axis);
        } else if (dtype->type == 'd') {
            // double (float64) array
            hash_axis<double>(inarray, result, axis);
        } else if (dtype->type == 'i') {
            // int32 array
            hash_axis<int>(inarray, result, axis);
        } else if (dtype->type == 'I') {
            // uint32 array
            hash_axis<unsigned int>(inarray, result, axis);
        } else if (dtype->type == 'l') {
            // int64 array
            hash_axis<long>(inarray, result, axis);
        } else if (dtype->type == 'L') {
            // uint64 array
            hash_axis<unsigned long>(inarray, result, axis);
        } else {
            PyErr_SetString(PyExc_TypeError, 
                    "Data array needs to be one of float, double, [u]int32, [u]int64");
            Py_DECREF(inarray);
            Py_DECREF(result);
            Py_RETURN_NONE;
        }

    } else {
        // Do element wise computations
        for (size_t i = 0; i < nd; ++i) {
            resultshape[i] = inshape[i];
        }
        
        result = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(resultdims, resultshape, NPY_UINT64));

        if (dtype->type == 'f') {
            // float32 array
            hash_elements<float>(inarray, result);
        } else if (dtype->type == 'd') {
            // double (float64) array
            hash_elements<double>(inarray, result);
        } else if (dtype->type == 'i') {
            // int32 array
            hash_elements<int>(inarray, result);
        } else if (dtype->type == 'I') {
            // uint32 array
            hash_elements<unsigned int>(inarray, result);
        } else if (dtype->type == 'l') {
            // int64 array
            hash_elements<long>(inarray, result);
        } else if (dtype->type == 'L') {
            // uint64 array
            hash_elements<unsigned long>(inarray, result);
        } else {
            PyErr_SetString(PyExc_TypeError, 
                    "Data array needs to be one of float, double, [u]int32, [u]int64");
            Py_DECREF(inarray);
            Py_DECREF(result);
            Py_RETURN_NONE;
        }
    }

    Py_DECREF(inarray);
    return reinterpret_cast<PyObject *>(result);
}


static PyMethodDef methods[] = {
    { "fnv1a", reinterpret_cast<PyCFunction>(pyfnv1a),
        METH_VARARGS | METH_KEYWORDS,
        "function docstring"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initnphash(void) {

    (void) Py_InitModule("nphash", methods);
    import_array();
}



