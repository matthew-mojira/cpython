import argparse
from analyzer import (Analysis, Instruction, analyze_files, Skip, Uop)
from generators_common import (ROOT, DEFAULT_INPUT, write_header, emit_tokens, replace_decrefs, replace_error, emit_to)
from py_metadata_generator import get_specialized
from cwriter import CWriter
from typing import TextIO, Iterator
from lexer import Token
from stack import Stack

DEFAULT_OUTPUT = ROOT / "Tools/cases_generator/output/wasm_cases.c"

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
    emit_tokens(out, uop, Stack(), inst, WASM_REPLACEMENTS)
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
        wat_out: TextIO,
        ) -> None:
    write_header(__file__, filenames, outfile)
    out = CWriter(outfile, 0, False)
    out_wat = CWriter(wat_out, 0, False)

    specialized = get_specialized(analysis)

    i = 0
    bytecodes = 0
    total = 0

    # emit struct definition for two values
    out.emit("\n")
    out.emit("struct two_values { PyObject *first; PyObject *second; };\n")

    for mnemonic, instruction in analysis.instructions.items():
        # skip all specialized opcodes
        if mnemonic in specialized:
            print("skipping specialized bytecode", mnemonic)
            continue
        if mnemonic.startswith("INSTRUMENTED"):
            print("skipping instrumented bytecode", mnemonic)
            continue

        # out.emit(f"{mnemonic}: {instruction.properties.tier}\n")
        props = instruction.properties

        # skip bad properties
        if props.escapes:
            print("skipping escaping bytecode", mnemonic)
            continue
        # skip large instructions
        #if len(instruction.parts) > 1:
        #    print("skipping multi-part bytecode", mnemonic)
        #    continue
        # skip jump instructions
        #if props.jumps:
        #    print("skipping jumping instruction", mnemonic)
        #    continue
        # skip "always exits" instructions
        #if props.always_exits:
        #    print('skipping "always exits" instruction', mnemonic)
        #    continue
        # skip "stores sp" instructions
        if props.stores_sp:
            print('skipping "stores sp" instruction', mnemonic)
            continue

        total += 1

        out.emit("\n/* ------------------------\n")
        out.emit(f" * OPCODE: {mnemonic}\n")

        # At current moment, no such instructions exist which fulfill the following properties:
        #   deopts = True       \
        #   side_exit = True    | all seem to be eliminated by removing specialized bytecodes
        #   tier != None        |
        #   oparg_and_1 = True  /
        #   const_oparg != -1
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
            out.emit(" * INCOMPATIBLE!\n")

            #continue

        out.emit(" * PROPERTIES:\n")
        for (key, value) in props.__dict__.items():
            out.emit(f" *   {key}: {value}\n")
        out.emit(" */\n")
        out.emit("// @@@!!\n")

        out.emit(f"// PARTS: {len(instruction.parts)}\n")

        for part in instruction.parts:
            # Uop or skip, assume Uop
            if isinstance(part, Skip):
                out.emit(f"// SKIP unused cache entry/{part.size}\n")
                continue
            if "specializing" in part.annotations:
                out.emit(f"// SKIP specializing: {mnemonic}\n")
                continue
            out.emit(f"// PART {part}\n")

            inputs = len(part.stack.inputs) + part.properties.oparg
            outputs = len(part.stack.outputs)

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
                    out.emit("// >>#!@#@! SKIPPING because there are more than 2 outputs\n")
                    continue
            # name
            decl += f"hander_{mnemonic}("

            # params
            params = [f"{'PyObject *' if input.type is None or input.type == "" else input.type}{input.name}" for input in part.stack.inputs]
            if part.properties.oparg:
                params.append("int oparg") # load const
            if part.properties.uses_frame:
                params.append("_PyInterpreterFrame *frame") # load global
            if part.properties.uses_tstate:
                params.append("PyThreadState *tstate") # load global

            # add params to decl
            decl += "void" if not params else ", ".join(params)

            # closing )
            decl += ")"

            # wasm import name
            wasm_import = f'(import "python" "hander_{mnemonic}" (func $handler_{mnemonic} (param{' i32' * len(params)}) (result{' i32' * outputs})))\n'
            out.emit(f"// {wasm_import}")
            out_wat.emit(wasm_import)
            out.emit("\n")

            # declaration + attribute
            # out.emit(f'__attribute__ ((export_name("hander_{mnemonic}")))\n')
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
    print("Failed to translate at least", bytecodes, "of", total, "bytecodes")


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
        with open("output/wasm_cases.wat", "w") as outfile_wasm:
            generate_wasm(args.input, data, outfile, outfile_wasm)
