(module
    (import "python" "wasm_get_python_consts" (func $wasm_get_python_consts (result i32)))
    (import "python" "Wasm_Load_Const" (func $Wasm_Load_Const (param i32 i32) (result i32)))
    (import "python" "Wasm_Get_Long" (func $Wasm_Get_Long (param i32) (result i32)))

    (func (export "run")  (result i32)
        (local $const_pool i32)
        call $wasm_get_python_consts
        local.set $const_pool
        (call $Wasm_Load_Const (local.get $const_pool) (i32.const 0))
        call $Wasm_Get_Long
    )
)