#include "Python.h"
#include "pycore_ceval.h"
#include "ceval_macros.h"

struct two_values { PyObject *first; PyObject *second; };

void do_nothing(void) {
    return;
}

#include "wasm_handlers.c.h"
