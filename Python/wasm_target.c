#ifndef Py_INTERNAL_WASM_H
#define Py_INTERNAL_WASM_H

#include "Python.h"
#include "pycore_pymem.h"
#include "wasm_target.h"

// WASM types
typedef struct _wasm_structure {
    enum { WASM_BLOCK, WASM_LOOP, WASM_IF, WASM_BR, WASM_RETURN, PY_WRAPPER, STRUCT_APPEND } w_type;
    union {
        struct _wasm_structure *w_one;
        struct {
            struct _wasm_structure *fst;
            struct _wasm_structure *snd;
        } w_two;
        void *w_block;
        int w_int;
        /* do not use this union for WASM_RETURN */
    } w_data;
} wasm_structure;


#define WASM_ALLOC() (PyMem_Calloc(1, sizeof(wasm_structure)))
#define WASM_SET_ONE(X) w->w_data.w_one = ( X );
#define WASM_SET_TWO(X, Y) w->w_data.w_two.fst = ( X ); w->w_data.w_two.snd = ( Y );

Wasm wasm_wrapper(void *data) {
    // PRINT_FUNC_1(data);

    Wasm w = WASM_ALLOC();
    w->w_type = PY_WRAPPER;
    w->w_data.w_block = data;
    return w;
}

Wasm wasm_block(Wasm b) {
    // PRINT_FUNC_1(b);

    Wasm w = WASM_ALLOC();
    w->w_type = WASM_BLOCK;
    WASM_SET_ONE(b);
    return w;
}

Wasm wasm_loop(Wasm b) {
    // PRINT_FUNC_1(b);

    Wasm w = WASM_ALLOC();
    w->w_type = WASM_LOOP;
    WASM_SET_ONE(b);
    return w;
}

Wasm wasm_if(Wasm t, Wasm f) {
    // PRINT_FUNC_2(t, f);

    Wasm w = WASM_ALLOC();
    w->w_type = WASM_IF;
    WASM_SET_TWO(t, f);
    return w;
}

Wasm wasm_br(int i) {
    // PRINT_FUNC_1(i);

    Wasm w = WASM_ALLOC();
    w->w_type = WASM_BR;
    w->w_data.w_int = i;
    return w;
}

Wasm wasm_return(void) {
    // PRINT_FUNC_0();

    Wasm w = WASM_ALLOC();
    w->w_type = WASM_RETURN;
    return w;
}

Wasm wasm_append(Wasm fst, Wasm snd) {
    // PRINT_FUNC_2(fst, snd);

    Wasm w = WASM_ALLOC();
    w->w_type = STRUCT_APPEND;
    WASM_SET_TWO(fst, snd);
    return w;
}

#define PRINT_INDENT(S) for (int i = 0; i < indent; i++) putchar(' '); printf("%s\n", S );

void _wasm_print(Wasm w, int indent) {
    // PRINT_FUNC_2(w, indent);

    // printf("%d: ", w->w_type);

    switch (w->w_type) {
    case WASM_BLOCK:
        // printf("WASM_BLOCK\n");
        PRINT_INDENT("block");
        _wasm_print(w->w_data.w_one, indent + 2);
        PRINT_INDENT("end");
        break;
    case WASM_LOOP:
        // printf("WASM_LOOP\n");
        PRINT_INDENT("loop");
        _wasm_print(w->w_data.w_one, indent + 2);
        PRINT_INDENT("end");
        break;
    case WASM_IF:
        // printf("WASM_IF\n");
        PRINT_INDENT("if");
        _wasm_print(w->w_data.w_two.fst, indent + 2);
        PRINT_INDENT("else");
        _wasm_print(w->w_data.w_two.snd, indent + 2);
        PRINT_INDENT("end");
        break;
    case WASM_BR:
        // printf("WASM_BR\n");
        for (int i = 0; i < indent; i++)
            putchar(' ');
        printf("br %d\n", w->w_data.w_int);
        break;
    case WASM_RETURN:
        // printf("WASM_RETURN\n");
        PRINT_INDENT("return");
        break;
    case PY_WRAPPER:
        // printf("PY_WRAPPER\n");
        /* print basic block */
        // basicblock *b = (basicblock *) w->w_data.w_block;
        PRINT_INDENT("<basic block>");
//         for (int i = 0; i < b->b_iused; i++) {
//             cfg_instr c = b->b_instr[i];
//             for (int i = 0; i < indent; i++)
//                 putchar(' ');
//             printf("py>%s %d", _PyOpcode_OpName[c.i_opcode], c.i_oparg);
//             if (is_jump(&c)) printf("(JUMP)");
//             printf("\n");
//         }
        break;
    case STRUCT_APPEND:
        // printf("STRUCT_APPEND\n");
        // printf("<<- fst\n");
        _wasm_print(w->w_data.w_two.fst, indent);
        // printf("<<- snd\n");
        _wasm_print(w->w_data.w_two.snd, indent);
        // printf("<<- end struct append\n");
        break;
    }
}

void wasm_print(Wasm w) {
    _wasm_print(w, 0);
}

#endif
