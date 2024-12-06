block2:
    LOAD_CONST 1
    BINARY_OP <BinaryOp.ADD: 0>
    COPY 1
    LOAD_CONST 10
    COMPARE_OP <Compare.LT_CAST: 16>
    POP_JUMP_IF_FALSE <block4>
    -> block3

(call $Wasm_Load_Const (local.get $const_pool) (i32.const 4))  ;; 1
i32.const 0   ;; ADD
call $Wasm_Binary_Op
call $dup
(call $Wasm_Load_Const (local.get $const_pool) (i32.const 3))  ;; 10
i32.const 16  ;; LT_CAST

