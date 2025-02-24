#ifndef Py_INTERNAL_CFG_H
#define Py_INTERNAL_CFG_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_compile.h"
#include "pycore_instruction_sequence.h"
#include "pycore_opcode_utils.h"

struct _PyCfgBuilder;

int _PyCfgBuilder_UseLabel(struct _PyCfgBuilder *g, _PyJumpTargetLabel lbl);
int _PyCfgBuilder_Addop(struct _PyCfgBuilder *g, int opcode, int oparg, _Py_SourceLocation loc);

struct _PyCfgBuilder* _PyCfgBuilder_New(void);
void _PyCfgBuilder_Free(struct _PyCfgBuilder *g);
int _PyCfgBuilder_CheckSize(struct _PyCfgBuilder* g);
int _PyCfgBuilder_GetSize(struct _PyCfgBuilder* g);

void _PyCfgBuilder_DebugPrint(struct _PyCfgBuilder *);
void _PyCfgBuilder_DebugPrintInstructionSequence(_PyInstructionSequence *seq);

void _PyCfgBuilder_ComputeDominators(struct _PyCfgBuilder *);
void _PyCfgBasicblock_ComputeImmediateDominators(struct _PyCfgBuilder *);
void _PyCfgBuilder_ReversePostorder(struct _PyCfgBuilder *);
void _PyCfgBasicblock_ComputeDominatorTree(struct _PyCfgBuilder *);

void _PyCfgBuilder_BeyondRelooper(struct _PyCfgBuilder *);

int _PyCfg_ResolveJumpsAndExceptions(struct _PyCfgBuilder *g);
int _PyCfg_OptimizeCodeUnit(struct _PyCfgBuilder *g, PyObject *consts, PyObject *const_cache,
                            int nlocals, int nparams, int firstlineno);

int _PyCfg_ToInstructionSequence(struct _PyCfgBuilder *g, _PyInstructionSequence *seq);
int _PyCfg_OptimizedCfgToInstructionSequence(struct _PyCfgBuilder *g, _PyCompile_CodeUnitMetadata *umd,
                                             int code_flags, int *stackdepth, int *nlocalsplus,
                                             _PyInstructionSequence *seq);

PyCodeObject *
_PyAssemble_MakeCodeObject(_PyCompile_CodeUnitMetadata *u, PyObject *const_cache,
                           PyObject *consts, int maxdepth, _PyInstructionSequence *instrs,
                           int nlocalsplus, int code_flags, PyObject *filename);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_CFG_H */
