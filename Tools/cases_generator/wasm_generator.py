import argparse
from analyzer import (Analysis, Instruction, analyze_files, Skip, Uop)
from generators_common import (ROOT, DEFAULT_INPUT, write_header, emit_tokens)
from cwriter import CWriter
from typing import TextIO
from stack import Stack

DEFAULT_OUTPUT = ROOT / "Tools/cases_generator/output/wasm_cases.c"

def write_uop(
        uop: Uop, out: CWriter, offset: int, inst: Instruction, braces: bool
        ) -> int:
    out.start_line()
    if braces:
        out.emit(f"// {uop.name}\n")
#     for var in reversed(uop.stack.inputs):
#         out.emit(stack.pop(var))
    if braces:
        out.emit("{\n")
#         if not uop.properties.stores_sp:
#             for i, var in enumerate(uop.stack.outputs):
#                 out.emit(stack.push(var))
#         for cache in uop.caches:
#             if cache.name != "unused":
#                 if cache.size == 4:
#                     type = "PyObject *"
#                     reader = "read_obj"
#                 else:
#                     type = f"uint{cache.size*16}_t "
#                     reader = f"read_u{cache.size*16}"
#                 out.emit(
#                     f"{type}{cache.name} = {reader}(&this_instr[{offset}].cache);\n"
#                 )
#                 if inst.family is None:
#                     out.emit(f"(void){cache.name};\n")
#             offset += cache.size
    emit_tokens(out, uop, Stack(), inst)
#         if uop.properties.stores_sp:
#             for i, var in enumerate(uop.stack.outputs):
#                 out.emit(stack.push(var))
    if braces:
        out.start_line()
        out.emit("}\n")
    # out.emit(stack.as_comment() + "\n")
    return offset

def generate_wasm(
        filenames: list[str],
        analysis: Analysis,
        outfile: TextIO,
        ) -> None:
    write_header(__file__, filenames, outfile)
    out = CWriter(outfile, 0, False)

    i = 0
    bytecodes = 0

    # emit struct definition for two values
    out.emit("\n")
    out.emit("struct two_values { PyObject *first; PyObject *second; };\n")

    for mnemonic, instruction in analysis.instructions.items():
        # out.emit(f"{mnemonic}: {instruction.properties.tier}\n")
        props = instruction.properties
        if any([props.escapes, props.error_with_pop, props.error_without_pop, props.deopts,
                #props.oparg,
                props.jumps, props.eval_breaker, props.ends_with_eval_breaker,
                props.needs_this, props.always_exits, props.stores_sp, props.uses_co_consts,
                props.uses_co_names,
                #props.uses_locals,
                not props.pure,
                props.has_free, props.side_exit,
                props.oparg_and_1, props.const_oparg != -1]):
            bytecodes += 1
            continue

        out.emit("\n/* ------------------------\n")
        out.emit(f" * OPCODE: {mnemonic}\n")
        out.emit(" * PROPERTIES:\n")
        for (key, value) in props.__dict__.items():
            out.emit(f" *   {key}: {value}\n")
        out.emit(" */\n")

        for part in instruction.parts:
            # Uop or skip, assume Uop
            if isinstance(part, Skip):
                continue

            inputs = len(part.stack.inputs) + part.properties.oparg
            outputs = len(part.stack.outputs)

            out.emit("\n")
            out.emit(f'// (import "python" "handler{part.name}" (func $handler{part.name} (param{' i32' * inputs}) (result{' i32' * outputs})))\n')
            out.emit("\n")

            decl = ""
            # output type
            match outputs:
                case 2:
                    decl += "struct two_values "
                case 1:
                    output = part.stack.outputs[0]
                    decl += "PyObject *" if output.type is None or output.type == "" else output.type
                case 0:
                    decl += "void "
                case _:
                    print("Skipping", mnemonic)
                    out.emit(">>#!@#@! SKIPPING because there are more than 2 outputs\n")
                    continue
            # name
            decl += f"handler{part.name}("
            # params
            if inputs == 0:
                decl += "void"
            else:
                if part.properties.oparg:
                    decl += "int oparg"
                    decl += "".join([f", {'PyObject *' if input.type is None or input.type == "" else input.type}{input.name}" for input in part.stack.inputs])
                else:
                    # not handling condition, (size, peek (?))
                    decl += ", ".join([f"{'PyObject *' if input.type is None or input.type == "" else input.type}{input.name}" for input in part.stack.inputs])

            # closing )
            decl += ")"

            # declaration + attribute
            out.emit(f'__attribute__ ((export_name("handler{part.name}")))\n')
            out.emit(f"{decl};\n")

            # definition
            out.emit("\n")
            out.emit(decl);
            out.emit(" {\n");
            # define return value if need one
            out.emit("// (matthew) begin emitting space for return\n")
            if outputs == 2:
                out.emit("struct two_values two_value_return;\n")
                for i in range(2):
                    output = part.stack.outputs[i]
                    if not any(e.name == output.name for e in part.stack.inputs):
                        # do not emit return variable declaration if it is already a parameter
                        out.emit(f'{"PyObject *" if output.type is None or output.type == "" else output.type}{output.name};\n')
            elif outputs == 1:
                output = part.stack.outputs[0]
                if not any(e.name == output.name for e in part.stack.inputs):
                    # do not emit return variable declaration if it is already a parameter
                    out.emit(f'{"PyObject *" if output.type is None or output.type == "" else output.type}{output.name};\n')
            out.emit("// (matthew) end emitting space for return\n\n")
            # body
            write_uop(part, out, 1, instruction, False)
            # return
            out.emit("\n\n")
            out.emit("// (matthew) begin return\n")
            if outputs == 1:
                output = part.stack.outputs[0]
                out.emit(f"return {output.name};\n")
            elif outputs == 2:
                out.emit(f"two_value_return.first = {part.stack.outputs[0].name};\n")
                out.emit(f"two_value_return.second = {part.stack.outputs[1].name};\n")
                out.emit("return two_value_return;\n")
            out.emit("}\n")

#         i += 1
#         if i > 2:
#             print("break")
#             break
    print("Failed to translate at least", bytecodes, "of", len(analysis.instructions), "bytecodes")


arg_parser = argparse.ArgumentParser(
    description="Generate WASM module.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

arg_parser.add_argument(
    "-o", "--output", type=str, help="Generated code", default=DEFAULT_OUTPUT
)

arg_parser.add_argument(
    "input", nargs=argparse.REMAINDER, help="Instruction definition file(s)"
)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    if len(args.input) == 0:
        args.input.append(DEFAULT_INPUT)
    data = analyze_files(args.input)
    with open(args.output, "w") as outfile:
        generate_wasm(args.input, data, outfile)
