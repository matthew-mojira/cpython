typedef Wasm /* TODO */;

Wasm wasm_block(Wasm);
Wasm wasm_loop(Wasm);
Wasm wasm_if(Wasm, Wasm);

Wasm wasm_br(int, Wasm);
Wasm wasm_return();

Wasm wasm_wrapper(/* TODO should be basic block */);

Wasm wasm_seq(Wasm, Wasm);


