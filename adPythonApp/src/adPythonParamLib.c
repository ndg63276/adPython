#include "adPythonPlugin.h"

static PyObject *adPythonParamLibError;

static PyObject *adPythonParamLib_createParam(PyObject *self, PyObject *args) {    
    const char *param;
    PyObject *ptr;
    int sts, typ, key;
    if (!PyArg_ParseTuple(args, "Ois", &ptr, &typ, &param)) {
        PyErr_SetString(adPythonParamLibError, "PyArg_ParseTuple failed");
        return NULL;
    }
    adPythonPlugin * plugin = (adPythonPlugin *) PyCapsule_GetPointer(ptr, "adPythonPlugin");
    sts = plugin->createParam(param, typ, &key);
    if (sts < 0) {
        PyErr_SetString(adPythonParamLibError, "createParam failed");
        return NULL;
    }
    return PyLong_FromLong(key);
}

static PyObject *adPythonParamLib_setParam(PyObject *self, PyObject *args) {    
    PyObject *ptr, *value;
    int sts, typ, key;
    if (!PyArg_ParseTuple(args, "OiiO", &ptr, &typ, &key, &value)) {
        PyErr_SetString(adPythonParamLibError, "PyArg_ParseTuple failed");
        return NULL;
    }
    adPythonPlugin * plugin = (adPythonPlugin *) PyCapsule_GetPointer(ptr, "adPythonPlugin");
    if (typ == asynParamFloat64) {
        if (!PyFloat_Check(value)) {
            PyErr_SetString(adPythonParamLibError, "Float required for this param");
            return NULL;
        }        
        sts = plugin->setDoubleParam(key, PyFloat_AsDouble(value));
    } else {
        if (!PyInt_Check(value)) {
            PyErr_SetString(adPythonParamLibError, "Int required for this param");
            return NULL;
        }  
        sts = plugin->setIntegerParam(key, PyInt_AsLong(value));
    }
    if (sts < 0) {
        PyErr_SetString(adPythonParamLibError, "setParam failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *adPythonParamLib_getParam(PyObject *self, PyObject *args) {    
    PyObject *ptr;
    int sts, typ, key;
    if (!PyArg_ParseTuple(args, "Oii", &ptr, &typ, &key)) {
        PyErr_SetString(adPythonParamLibError, "PyArg_ParseTuple failed");
        return NULL;
    }
    adPythonPlugin * plugin = (adPythonPlugin *) PyCapsule_GetPointer(ptr, "adPythonPlugin");
    if (typ == asynParamFloat64) {
        double value;
        sts = plugin->getDoubleParam(key, &value);
        if (sts < 0) {
            PyErr_SetString(adPythonParamLibError, "getDoubleParam failed");
            return NULL;
        }
        return PyFloat_FromDouble(value);
    } else {
        long value;
        sts = plugin->getIntegerParam(key, &value);
        if (sts < 0) {
            PyErr_SetString(adPythonParamLibError, "getIntegerParam failed");
            return NULL;
        }
        return PyLong_FromLong(value);
    }
}

static PyMethodDef adPythonParamLibMethods[] = {
    {"createParam",  adPythonParamLib_createParam, METH_VARARGS,
     "Create a param."},
    {"setParam",  adPythonParamLib_setParam, METH_VARARGS,
     "Set a param."},  
    {"getParam",  adPythonParamLib_getParam, METH_VARARGS,
     "Get a param."},    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initadPythonParamLib(void)
{
    PyObject *m;

    m = Py_InitModule("adPythonParamLib", adPythonParamLibMethods);
    if (m == NULL)
        return;

    adPythonParamLibError = PyErr_NewException((char *)"adPythonParamLib.error", NULL, NULL);
    Py_INCREF(adPythonParamLibError);
    PyModule_AddObject(m, "error", adPythonParamLibError);
}

