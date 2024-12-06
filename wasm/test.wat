(module
    (import "python" "wasm_get_python_consts" (func $wasm_get_python_consts (result i32)))
    (import "python" "Wasm_Load_Const" (func $Wasm_Load_Const (param i32 i32) (result i32)))
    (import "python" "Wasm_Get_Long" (func $Wasm_Get_Long (param i32) (result i32)))
    (import "python" "Wasm_Binary_Comp" (func $Wasm_Binary_Comp (param i32 i32 i32) (result i32)))
    (import "python" "Wasm_Binary_Op" (func $Wasm_Binary_Op (param i32 i32 i32) (result i32)))
    (import "python" "PyObject_IsTrue" (func $PyObject_IsTrue (param i32) (result i32)))
    (import "python" "debug_print_here" (func $here (param i32)))

    (func $dup (param i32) (result i32 i32)
        local.get 0
        local.get 0)


    (func (export "run")  (result i32)
        (local $const_pool i32)
        call $wasm_get_python_consts
        local.set $const_pool
        (block (result i32 i32)
            (call $Wasm_Load_Const (local.get $const_pool) (i32.const 0))
            (call $Wasm_Load_Const (local.get $const_pool) (i32.const 1))
            call $dup
            (call $Wasm_Load_Const (local.get $const_pool) (i32.const 2))
            i32.const 16
            call $Wasm_Binary_Comp
            call $PyObject_IsTrue
            (call $here (i32.const 0))
            (if (param i32 i32)
                (then 
                    (loop (param i32 i32)
                        (call $Wasm_Load_Const (local.get $const_pool) (i32.const 3))  ;; 1
                        i32.const 0   ;; ADD
                        call $Wasm_Binary_Op
                        call $dup
                        (call $Wasm_Load_Const (local.get $const_pool) (i32.const 2))  ;; 10
                        i32.const 16  ;; LT_CAST
                        call $Wasm_Binary_Comp
                        call $PyObject_IsTrue
                        (if (param i32 i32)
                            (then (br 1))
                            (else (br 3)))
                        unreachable))
                (else (br 1)))
            unreachable)
        (call $here (i32.const 2))
        i32.const 0   ;; ADD
        call $Wasm_Binary_Op
        call $Wasm_Get_Long
        return 
    )
)