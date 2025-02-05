import argparse
from analyzer import (Analysis, analyze_files, Skip, Uop)
from generators_common import (ROOT, DEFAULT_INPUT, write_header)
from tier1_generator import write_uop
from cwriter import CWriter
from typing import TextIO
from stack import Stack

DEFAULT_OUTPUT = ROOT / "Tools/cases_generator/output/wasm_cases.wat"

def generate_wasm(
        filenames: list[str],
        analysis: Analysis,
        outfile: TextIO,
        ) -> None:
    write_header(__file__, filenames, outfile)
    out = CWriter(outfile, 0, False)

    i = 0

    for mnemonic, instruction in analysis.instructions.items():
        if mnemonic != "NOP" and mnemonic != "LOAD_FAST" and mnemonic != "STORE_FAST" and mnemonic != "TO_BOOL":
            continue
        # out.emit(f"{mnemonic}: {instruction.properties.tier}\n")

        for part in instruction.parts:
            # Uop or skip, assume Uop
            if isinstance(part, Uop):
                inputs = len(part.stack.inputs)
                outputs = len(part.stack.outputs)
                out.emit("\nWasm import\n\n")
                out.emit(f'(import "python" "handler{part.name}" (func $handler{part.name} (param{' i32' * inputs}) (result{' i32' * outputs})))\n')

                out.emit("\nC function\n\n")
                decl = ""
                # output type
                if outputs > 1:
                    fail
                if outputs == 0:
                    decl += "void "
                else:
                    output = part.stack.outputs[0]
                    decl += "PyObject *" if output.type is None or output.type == "" else output.type
                # name
                decl += f"handler{part.name}("
                # params
                if inputs == 0:
                    decl += "void"
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
                if outputs == 1:
                    output = part.stack.outputs[0]
                    out.emit(f'{"PyObject *" if output.type is None or output.type == "" else output.type}{output.name};\n\n')
                # body
                write_uop(part, out, 1, Stack(), instruction, False)
                # return
                out.emit("\n")
                if outputs == 1:
                    out.emit("\n")
                    output = part.stack.outputs[0]
                    out.emit(f"return {output.name};\n")
                out.emit("}\n")

#         i += 1
#         if i > 2:
#             print("break")
#             break


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
