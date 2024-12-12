(module
    (import "python" "wasm_get_python_consts" (func $wasm_get_python_consts (result i32)))
    (import "python" "Wasm_Load_Const" (func $Wasm_Load_Const (param i32 i32) (result i32)))
    (import "python" "Wasm_From_Long" (func $Wasm_From_Long (param i32) (result i32)))
    (import "python" "Wasm_Get_Long" (func $Wasm_Get_Long (param i32) (result i32)))
    (import "python" "Wasm_Binary_Comp" (func $Wasm_Binary_Comp (param i32 i32 i32) (result i32)))
    (import "python" "Wasm_Binary_Op" (func $Wasm_Binary_Op (param i32 i32 i32) (result i32)))
    (import "python" "PyObject_IsTrue" (func $PyObject_IsTrue (param i32) (result i32)))
    (import "python" "debug_print_here" (func $here (param i32)))
    (import "python" "Wasm_PyObject_ToBool" (func $Wasm_PyObject_ToBool (param i32) (result i32)))
    (import "python" "Wasm_IncRef" (func $Wasm_IncRef (param i32)))
    (import "python" "Wasm_DecRef" (func $Wasm_DecRef (param i32)))

    (func $dup (param i32) (result i32 i32)
        local.get 0
        local.get 0)
   (func (export "run") (param $py_0_int i32) (result i32)
      local.get $py_0_int
      call $Wasm_From_Long
      call $foo
      call $Wasm_Get_Long
   )
   (func $foo (param $py_0 i32) (result i32)
      (local $py_1 i32)
      (local $const_pool i32)
      call $wasm_get_python_consts
      local.set $const_pool
      block (param ) (result )
         local.get $const_pool
         i32.const 1
         call $Wasm_Load_Const
         call $dup
         call $Wasm_DecRef
         local.set $py_1
         local.get $py_1
         call $dup
         call $Wasm_IncRef
         local.get $const_pool
         i32.const 2
         call $Wasm_Load_Const
         i32.const 148
         call $Wasm_Binary_Comp
         call $PyObject_IsTrue
         if (param ) (result )
            loop (param ) (result )
               local.get $py_1
               call $dup
               call $Wasm_IncRef
               local.get $const_pool
               i32.const 3
               call $Wasm_Load_Const
               i32.const 10
               call $Wasm_Binary_Op
               call $dup
               call $Wasm_DecRef
               local.set $py_1
               local.get $py_1
               call $dup
               call $Wasm_IncRef
               local.get $const_pool
               i32.const 2
               call $Wasm_Load_Const
               i32.const 148
               call $Wasm_Binary_Comp
               call $PyObject_IsTrue
               if (param ) (result )
                  br 1
               else
                  br 3
               end
               unreachable
            end
         else
            br 1
         end
         unreachable
      end
      local.get $py_0
      call $dup
      call $Wasm_IncRef
      return
   )
)
