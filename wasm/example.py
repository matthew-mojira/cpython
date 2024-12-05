
def f(a,b):
    if a < 0:
        return b * 3 + a
    else:
        while a > 0:
            b = b +a
            a = a - 1
    return f"hello {b}"
def f1(b):
    a = 10
    if b > 5:
        return a +1 
    else:
        return a +2 

def f2():
    b = 4
    a = 3
    while a < 10:
        a = a + 1
    return a + b

import dis 
dis.dis(f2)

print(f2())


from bytecode import (
    Instr,
    Label,
    Bytecode,
    Compare,
    BinaryOp,
    dump_bytecode,
    ControlFlowGraph,
    BasicBlock,
)

# L1 = Label()
# code = Bytecode(
#     [
#         Instr("RESUME", 0),
#         Instr("LOAD_FAST", "a"),
#         Instr("LOAD_CONST", 10),
#         Instr("LOAD_CONST", 5),
#         Instr("COMPARE_OP", Compare.GT_CAST),
#         Instr("POP_JUMP_IF_FALSE", L1),

#         Instr("LOAD_CONST", 1),
#         Instr("BINARY_OP", BinaryOp.ADD),
#         Instr("RETURN_VALUE"),
#         # Label L1
#         L1,
#         Instr("LOAD_CONST", 2),
#         Instr("BINARY_OP", BinaryOp.ADD),
#         Instr("RETURN_VALUE"),
#     ]
# )
L1 = Label()
L2 = Label()
code = Bytecode(
    [
        Instr("RESUME", 0),
        
        Instr("LOAD_CONST", 4),
        Instr("LOAD_CONST", 3), 
        Instr("COPY", 1),
        Instr("LOAD_CONST", 10), 
        Instr("COMPARE_OP", Compare.LT_CAST), 
        Instr("POP_JUMP_IF_FALSE", L2),

        # Label L1
        L1,
        Instr("LOAD_CONST", 1),  # 1
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("COPY", 1),
        Instr("LOAD_CONST", 10), 
        Instr("COMPARE_OP", Compare.LT_CAST),  # bool(<)
        Instr("POP_JUMP_IF_FALSE", L2),
        Instr("JUMP_BACKWARD", L1),
        
        # Label L2
        L2,
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("RETURN_VALUE"),
    ]
)

def k():
    
    pass 

# code.legalize()
# dump_bytecode(code)
# code.argcount = 1
k.__code__ = code.to_code()

print(k())
import bytecode
# 
# bytecode.dump_bytecode(bytecode.Bytecode.from_code(k.__code__))

# print(Instr("STORE_FAST", "a").stack_effect())