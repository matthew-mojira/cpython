from wasmtime import Engine, Store, Module, Linker, WasiConfig, Config, Instance

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
wasi.preopen_dir("/home/mschnei2/cpython", "/")

store.set_wasi(wasi)

python_wasm_instance = linker.instantiate(store, python_wasm)
linker.define_instance(store, "python", python_wasm_instance)
main_instance = linker.instantiate(store, external_mod)
print(main_instance.exports(store)['run'](store))
