#include "Python.h"
#include "pycore_ceval.h"
#include "ceval_macros.h"

void do_nothing(void) {
    return;
}

#include "wasm_cases.c.h"
