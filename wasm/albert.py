import subprocess
import os
from wasmtime import Engine, Store, Module, Linker, WasiConfig, Config, Instance

subprocess.run(['python',"-m", "compileall",'wasm/example.py']) 
subprocess.run(['mv', "wasm/__pycache__/example.cpython-313.pyc", "wasm/example.cpython-313.pyc"])
wat_output = subprocess.run(['python', 'wasm/beyond-relooper.py'], capture_output=True, text=True).stdout
with open("wasm/test.wat", "w") as wat_file:
    wat_file.write(wat_output)

# this was added to the _config.py
# @setter_property
#     def max_wasm_stack(self, size: int) -> None:
#         if not isinstance(size, int):
#             raise TypeError('expected a int')
#         ffi.wasmtime_config_max_wasm_stack_set(self.ptr(), size)
config = Config()
config.max_wasm_stack = 8388608
engine = Engine(config)
store = Store(engine)

# Load and compile our two modules
python_wasm = Module.from_file(engine, "cross-build/wasm32-wasip1/python.wasm")
external_mod = Module.from_file(engine, "wasm/test.wat")

linker = Linker(engine)
linker.define_wasi()
wasi = WasiConfig()
wasi.inherit_stdout()
wasi.preopen_dir(os.getcwd(), "/")

store.set_wasi(wasi)

python_wasm_instance = linker.instantiate(store, python_wasm)
linker.define_instance(store, "python", python_wasm_instance)
main_instance = linker.instantiate(store, external_mod)
print(main_instance.exports(store)['run'](store,int(input())))
