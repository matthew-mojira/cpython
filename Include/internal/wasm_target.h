struct _wasm_structure;
typedef struct _wasm_structure *Wasm;

Wasm wasm_block(Wasm);
Wasm wasm_loop(Wasm);
Wasm wasm_if(Wasm, Wasm);

Wasm wasm_br(int);
Wasm wasm_return(void);

Wasm wasm_wrapper(void *);

Wasm wasm_append(Wasm, Wasm);

void wasm_print(Wasm);
