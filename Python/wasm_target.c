#ifndef Py_INTERNAL_WASM_H
#define Py_INTERNAL_WASM_H

#include "Python.h"
#include "pycore_flowgraph.h"
#include "pycore_pymem.h"
#include "wasm_target.h"

// WASM types
typedef struct _wasm_structure {
    enum {
        WASM_BLOCK,
        WASM_LOOP,
        WASM_IF,
        WASM_BR,
        WASM_RETURN,
        PY_WRAPPER,
        STRUCT_APPEND,
        TEMP_STRING,
        NULL_NULL
    } w_type;
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

Wasm wasm_string(const char *str) {
    Wasm w = WASM_ALLOC();
    w->w_type = TEMP_STRING;
    w->w_data.w_block = str;
    return w;
}

Wasm wasm_null(void) {
    Wasm w = WASM_ALLOC();
    w->w_type = NULL_NULL;
    return w;
}

Wasm wasm_string0(void) {
    return wasm_null();
}

Wasm wasm_string1(const char *str1) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string0());
}

Wasm wasm_string2(const char *str1, const char *str2) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string1(str2));
}

Wasm wasm_string3(const char *str1, const char *str2, const char *str3) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string2(str2, str3));
}

Wasm wasm_string4(const char *str1, const char *str2, const char *str3, const char *str4) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string3(str2, str3, str4));
}

Wasm wasm_string5(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string4(str2, str3, str4, str5));
}

Wasm wasm_string6(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5, const char *str6) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string5(str2, str3, str4, str5, str6));
}

Wasm wasm_string7(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5, const char *str6, const char *str7) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string6(str2, str3, str4, str5, str6, str7));
}

Wasm wasm_string8(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5, const char *str6, const char *str7, const char *str8) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string7(str2, str3, str4, str5, str6, str7, str8));
}

Wasm wasm_string9(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5, const char *str6, const char *str7, const char *str8, const char *str9) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string8(str2, str3, str4, str5, str6, str7, str8, str9));
}

Wasm wasm_string10(const char *str1, const char *str2, const char *str3, const char *str4, const char *str5, const char *str6, const char *str7, const char *str8, const char *str9, const char *str10) {
    Wasm w = wasm_string(str1);
    return wasm_append(w, wasm_string9(str2, str3, str4, str5, str6, str7, str8, str9, str10));
}

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
#define PURPLE  "\033[1;35m"
#define CYAN    "\033[1;36m"
#define WHITE   "\033[1;37m"

#define BG_RED    "\033[41m"
#define BG_GREEN  "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE   "\033[44m"
#define BG_PURPLE "\033[45m"
#define BG_CYAN   "\033[46m"
#define BG_WHITE  "\033[47m"

#define INDENT 4
#define PRINT_INDENT(S, C) for (int i = 0; i < indent; i++) putchar(' '); printf("%s%s%s\n", C , S , RESET);

void _wasm_print(Wasm w, int indent) {
    // PRINT_FUNC_2(w, indent);

    // printf("%d: ", w->w_type);

    switch (w->w_type) {
    case WASM_BLOCK:
        // printf("WASM_BLOCK\n");
        PRINT_INDENT("block", YELLOW);
        _wasm_print(w->w_data.w_one, indent + INDENT);
        PRINT_INDENT("end", YELLOW);
        break;
    case WASM_LOOP:
        // printf("WASM_LOOP\n");
        PRINT_INDENT("loop", YELLOW);
        _wasm_print(w->w_data.w_one, indent + INDENT);
        PRINT_INDENT("end", YELLOW);
        break;
    case WASM_IF:
        // printf("WASM_IF\n");
        PRINT_INDENT("if", YELLOW);
        _wasm_print(w->w_data.w_two.fst, indent + INDENT);
        PRINT_INDENT("else", YELLOW);
        _wasm_print(w->w_data.w_two.snd, indent + INDENT);
        PRINT_INDENT("end", YELLOW);
        break;
    case WASM_BR:
        // printf("WASM_BR\n");
        for (int i = 0; i < indent; i++)
            putchar(' ');
        printf("%sbr %d %s\n", RED, w->w_data.w_int, RESET);
        break;
    case WASM_RETURN:
        // printf("WASM_RETURN\n");
        PRINT_INDENT("return", RED);
        break;
    case PY_WRAPPER:
        // printf("PY_WRAPPER\n");
        /* print basic block */
        // basicblock *b = (basicblock *) w->w_data.w_block;
        PRINT_INDENT("<basic block>", BG_GREEN);
        _PyCfg_WasmPrintBasicBlock(w->w_data.w_block, indent);
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
    case TEMP_STRING:
        const char *const str = w->w_data.w_block;
        switch (str[0]) {
        case 'i':
            PRINT_INDENT(str, BLUE);
            break;
        case 'l':
            PRINT_INDENT(str, CYAN);
            break;
        case 'c':
            PRINT_INDENT(str, GREEN);
            break;
        default:
        }
        break;
    case NULL_NULL:
        break;
    }
}

void wasm_print(Wasm w) {
    printf(PURPLE"(func (param $tstate i32) (param $frame i32)"RESET"\n");
    _wasm_print(w, INDENT);
    printf(PURPLE")"RESET"\n");
}

#endif
