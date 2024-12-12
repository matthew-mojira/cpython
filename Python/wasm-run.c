#include "Python.h"
#include "pycore_abstract.h"      // _PyIndex_Check()
#include "pycore_backoff.h"
#include "pycore_call.h"          // _PyObject_CallNoArgs()
#include "pycore_cell.h"          // PyCell_GetRef()
#include "pycore_ceval.h"
#include "pycore_code.h"
#include "pycore_emscripten_signal.h"  // _Py_CHECK_EMSCRIPTEN_SIGNALS
#include "pycore_function.h"
#include "pycore_instruments.h"
#include "pycore_intrinsics.h"
#include "pycore_jit.h"
#include "pycore_long.h"          // _PyLong_GetZero()
#include "pycore_moduleobject.h"  // PyModuleObject
#include "pycore_object.h"        // _PyObject_GC_TRACK()
#include "pycore_opcode_metadata.h" // EXTRA_CASES
#include "pycore_optimizer.h"     // _PyUOpExecutor_Type
#include "pycore_opcode_utils.h"  // MAKE_FUNCTION_*
#include "pycore_pyatomic_ft_wrappers.h" // FT_ATOMIC_*
#include "pycore_pyerrors.h"      // _PyErr_GetRaisedException()
#include "pycore_pystate.h"       // _PyInterpreterState_GET()
#include "pycore_range.h"         // _PyRangeIterObject
#include "pycore_setobject.h"     // _PySet_Update()
#include "pycore_sliceobject.h"   // _PyBuildSlice_ConsumeRefs
#include "pycore_sysmodule.h"     // _PySys_Audit()
#include "pycore_tuple.h"         // _PyTuple_ITEMS()
#include "pycore_typeobject.h"    // _PySuper_Lookup()
#include "pycore_uop_ids.h"       // Uops
#include "pycore_pyerrors.h"

#include "pycore_dict.h"
#include "dictobject.h"
#include "pycore_frame.h"
#include "frameobject.h"          // _PyInterpreterFrame_GetLine
#include "opcode.h"
#include "pydtrace.h"
#include "setobject.h"
#include "ceval_macros.h"
#include "longobject.h"


PyObject* Wasm_Load_Const(PyObject* const_pool, int oparg){
    PyObject *value = GETITEM(const_pool, oparg);
    Py_INCREF(value);
    return value;
}



PyObject* Wasm_Binary_Op(PyObject* lhs, PyObject* rhs, int oparg)
{
    PyObject *res;
    res = _PyEval_BinaryOps[oparg](lhs, rhs);
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    return res;
}

PyObject* Wasm_Binary_Comp(PyObject* left, PyObject* right, int oparg)
{
    assert((oparg >> 5) <= Py_GE);
    int lhs = Wasm_Get_Long(left);
    int rhs = Wasm_Get_Long(right);

    PyObject* res = PyObject_RichCompare(left, right, oparg >> 5);
    Py_DECREF(left);
    Py_DECREF(right);
    if (oparg & 16) {
        int res_bool = PyObject_IsTrue(res);
        Py_DECREF(res);
        // if (res_bool < 0) goto pop_2_error;
        res = res_bool ? Py_True : Py_False;
    } 
    return res;
}


PyObject* Wasm_From_Long(int number)
{
    return PyLong_FromLong(number);
}


int Wasm_Get_Long(PyObject* obj)
{
    return PyLong_AsLong(obj);
}

void debug_print_here(int num)
{
    printf("%d \n", num);
    fflush(stdout);
}

PyObject* Wasm_PyObject_ToBool(PyObject* obj)
{
    int res_bool = PyObject_IsTrue(obj);
    Py_DECREF(obj);
    PyObject* res  = res_bool ? Py_True : Py_False;
    return res;
}


void Wasm_IncRef(obj)
PyObject *obj;
{
    Py_INCREF(obj);
}

void Wasm_DecRef(obj)
PyObject *obj;
{
    Py_DECREF(obj);
}