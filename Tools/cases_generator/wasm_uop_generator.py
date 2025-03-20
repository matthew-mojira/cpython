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

DEFAULT_OUTPUT = ROOT / "Tools/cases_generator/output/wasm_handlers.c"

def replace_error_with_assert(
        out: CWriter,
        tkn: Token,
        tkn_iter: Iterator[Token],
        uop: Uop,
        stack: Stack,
        inst: Instruction | None
        ) -> None:
    out.emit_at("assert(!", tkn)
    out.emit(next(tkn_iter))
    emit_to(out, tkn_iter, "COMMA")
    label = next(tkn_iter).text
    next(tkn_iter)  # RPAREN
    next(tkn_iter)  # Semi colon
    out.emit(")); // (matthew) replace error with assertion\n")
    if label != "error":
        print("label was not error:", uop.name)
    #out.emit(label)

def replace_error_no_pop(
    out: CWriter,
    tkn: Token,
    tkn_iter: Iterator[Token],
    uop: Uop,
    stack: Stack,
    inst: Instruction | None,
) -> None:
    next(tkn_iter)  # LPAREN
    next(tkn_iter)  # RPAREN
    next(tkn_iter)  # Semi colon
    out.emit_at("assert(0); // (matthew) replace error", tkn)

WASM_REPLACEMENTS = {
        "DECREF_INPUTS": replace_decrefs,
        "ERROR_IF": replace_error_with_assert,
        "ERROR_NO_POP": replace_error_no_pop,
        }

def write_uop(
        uop: Uop, out: CWriter, offset: int, inst: Instruction, braces: bool
        ) -> None:
    out.start_line()

    out.emit("\n/* ------------------------\n")
    out.emit(f" * UOP: {uop.name}\n")
    out.emit(" * PROPERTIES:\n")
    for (key, value) in uop.properties.__dict__.items():
        out.emit(f" *   {key}: {value}\n")
    out.emit(" */\n")

    inputs = len(uop.stack.inputs) + uop.properties.oparg
    outputs = len(uop.stack.outputs)

    out.emit("\n")

    decl = ""
    # output type
    match outputs:
        case 2:
            decl += "struct two_values "
        case 1:
            output = uop.stack.outputs[0]
            decl += "PyObject *" if output.type is None or output.type == "" else output.type
        case 0:
            decl += "void "
        case _:
            out.emit("// >>#!@#@! SKIPPING because there are more than 2 outputs\n")
            return
    # name
    decl += f"handler_{uop.name}("

    # params
    params = [f"{'PyObject *' if input.type is None or input.type == "" else input.type}{input.name}" for input in uop.stack.inputs]
    if uop.properties.oparg:
        params.append("int oparg") # load const
    if uop.properties.uses_frame:
        params.append("_PyInterpreterFrame *frame") # load global
    if uop.properties.uses_tstate:
        params.append("PyThreadState *tstate") # load global

    # add params to decl
    decl += "void" if not params else ", ".join(params)

    # closing )
    decl += ")"

    # wasm import name
    wasm_import = f'(import "python" "handler_{uop.name}" (func $handler_{uop.name} (param{' i32' * len(params)}) (result{' i32' * outputs})))\n'
    out.emit(f"// {wasm_import}")
    out.emit("\n")

    # declaration + attribute
    # attribute is only for emscripten?
    #out.emit(f'__attribute__ ((export_name("handler_{uop.name}")))\n')
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
            output = uop.stack.outputs[i]
            if not any(e.name == output.name for e in uop.stack.inputs):
                # do not emit return variable declaration if it is already a parameter
                out.emit(f'{"PyObject *" if output.type is None or output.type == "" else output.type}{output.name};\n')
    elif outputs == 1:
        output = uop.stack.outputs[0]
        if not any(e.name == output.name for e in uop.stack.inputs):
            # do not emit return variable declaration if it is already a parameter
            out.emit(f'{"PyObject *" if output.type is None or output.type == "" else output.type}{output.name};\n')
    out.emit("// (matthew) end emitting space for return\n\n")
    # body
    emit_tokens(out, uop, Stack(), inst, WASM_REPLACEMENTS)
    # return
    out.emit("\n\n")
    out.emit("// (matthew) begin return\n")
    if outputs == 1:
        output = uop.stack.outputs[0]
        out.emit(f"return {output.name};\n")
    elif outputs == 2:
        out.emit(f"two_value_return.first = {uop.stack.outputs[0].name};\n")
        out.emit(f"two_value_return.second = {uop.stack.outputs[1].name};\n")
        out.emit("return two_value_return;\n")
    out.emit("}\n")



def generate_wasm(
        filenames: list[str],
        analysis: Analysis,
        outfile: TextIO,
        ) -> None:
    write_header(__file__, filenames, outfile)
    out = CWriter(outfile, 0, False)

    #out.emit("\nstruct two_values { PyObject *first; PyObject *second; };\n")

    #out.emit("\n// INSTRUCTIONS:\n")

    # get uops only needed for supported instructions
    specialized = get_specialized(analysis)
    uops = []
    for mnemonic, instruction in analysis.instructions.items():
        if mnemonic in specialized or mnemonic.startswith("INSTRUMENTED"):
            continue
        props = instruction.properties
        if props.escapes or props.jumps or props.always_exits or props.stores_sp:
            continue

        #out.emit(f"// {mnemonic}:\n")

        parts = list(filter(lambda part: not isinstance(part, Skip) and "specializing" not in part.annotations, instruction.parts))
        for part in parts:
            #out.emit(f"//     {part.name}\n")
            if part not in uops:
                uops.append(part)

    out.emit("\n// MICRO-OPS:\n")
    for uop in uops:
        out.emit(f"// {uop.name}\n")

    # generate the uops individually first
    for uop in uops:
        write_uop(uop, out, 0, None, False)


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
