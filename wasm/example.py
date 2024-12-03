
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


import dis 
dis.dis(f1)
print(dis.cmp_op)

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

L1 = Label()
code = Bytecode(
    [
        Instr("RESUME", 0),
        Instr("LOAD_CONST", 10),
        Instr("LOAD_CONST", 10),
        Instr("LOAD_CONST", 5),
        Instr("COMPARE_OP", Compare.GT_CAST),
        Instr("POP_JUMP_IF_FALSE", L1),

        Instr("LOAD_CONST", 1),
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("RETURN_VALUE"),
        # Label L1
        L1,
        Instr("LOAD_CONST", 2),
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("RETURN_VALUE"),
    ]
)

code.legalize()
dump_bytecode(code)

print(exec(code.to_code()))
import bytecode

# bytecode.dump_bytecode(bytecode.Bytecode.from_code(f1.__code__))