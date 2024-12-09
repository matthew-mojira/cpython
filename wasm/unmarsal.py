import sys
import dis, marshal
# pyc_path = sys.argv[1]
with open("wasm/example.cpython-313.pyc", 'rb') as f:
    # First 16 bytes comprise the pyc header (python 3.6+), else 8 bytes.
    pyc_header = f.read(16)
    code_obj = marshal.load(f) # Suite to code object

dis.dis(code_obj)
print(dis.Bytecode(code_obj).info() )

for fn in code_obj.co_consts:
    print()
    print()
    print()
    try:
        print(dis.Bytecode(fn).info())
    except:
        pass