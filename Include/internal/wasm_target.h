struct _wasm_structure;
typedef struct _wasm_structure *Wasm;

Wasm wasm_block(Wasm);
Wasm wasm_loop(Wasm);
Wasm wasm_if(Wasm, Wasm);

Wasm wasm_br(int);
Wasm wasm_return(void);

Wasm wasm_wrapper(void *);

Wasm wasm_append(Wasm, Wasm);
Wasm wasm_null(void);

Wasm wasm_string0(void);
Wasm wasm_string1(const char *);
Wasm wasm_string2(const char *, const char *);
Wasm wasm_string3(const char *, const char *, const char *);
Wasm wasm_string4(const char *, const char *, const char *, const char *);
Wasm wasm_string5(const char *, const char *, const char *, const char *, const char *);
Wasm wasm_string6(const char *, const char *, const char *, const char *, const char *, const char *);
Wasm wasm_string7(const char *, const char *, const char *, const char *, const char *, const char *, const char *);
Wasm wasm_string8(const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *);
Wasm wasm_string9(const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *);
Wasm wasm_string10(const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *, const char *);

void wasm_print(Wasm);
