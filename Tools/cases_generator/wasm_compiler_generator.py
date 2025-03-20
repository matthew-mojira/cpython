import argparse
from analyzer import (Analysis, Instruction, analyze_files, Skip, Uop)
from generators_common import (ROOT, DEFAULT_INPUT, write_header, emit_tokens, replace_decrefs, replace_error, emit_to)
from py_metadata_generator import get_specialized
from cwriter import CWriter
from typing import TextIO, Iterator
from lexer import Token
from stack import Stack

"""
This generates the handlers (i.e. individual C functions) which is called upon
by generated wasm bytecode.
"""

DEFAULT_OUTPUT = ROOT / "Tools/cases_generator/output/wasm_cases.c.h"


def generate_wasm(
        filenames: list[str],
        analysis: Analysis,
        outfile: TextIO,
        ) -> None:
    write_header(__file__, filenames, outfile)
    out = CWriter(outfile, 0, False)

    out.emit("\n// switch (op) {\n")

    # get uops only needed for supported instructions
    specialized = get_specialized(analysis)
    for mnemonic, instruction in analysis.instructions.items():
        if mnemonic in specialized or mnemonic.startswith("INSTRUMENTED"):
            continue
        props = instruction.properties
        if props.escapes or props.jumps or props.always_exits or props.stores_sp:
            pass
            #continue

        out.emit(f"case {mnemonic}:\n")
        args = []

        parts = list(filter(lambda part: not isinstance(part, Skip) and "specializing" not in part.annotations, instruction.parts))
        for uop in parts:
            # set up args
            out.emit(f"    // real args: {len(uop.stack.inputs)}\n")
            if uop.properties.oparg:
                out.emit("    // i32.const <oparg>\n")
                args.append('"i32.const <oparg>"')
            if uop.properties.uses_frame:
                out.emit("    // local.get $frame\n")
                args.append('"local.get $frame"')
            if uop.properties.uses_tstate:
                out.emit("    // local.get $tstate\n")
                args.append('"local.get $tstate"')

            # call handler
            out.emit(f"    // call $handler_{uop.name}\n")
            args.append(f'"call $handler_{uop.name}"')

        # real C code
        out.emit(f'    wasm_instr = wasm_string{len(args)}({", ".join(args)});\n')
        out.emit("    break;\n")

    # out.emit("}\n")


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
    print(args.output)
    with open(args.output, "w") as outfile:
        generate_wasm(args.input, data, outfile)
