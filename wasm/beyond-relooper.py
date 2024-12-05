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

from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx

def cfg_to_network(cfg_bytecode : ControlFlowGraph):
    G = nx.DiGraph()
    for basic_block in cfg_bytecode:
        G.add_node(cfg_bytecode.get_block_index(basic_block), block=basic_block)

    for basic_block in cfg_bytecode:
        curr_index = cfg_bytecode.get_block_index(basic_block)
        jump_targets = []
        next_jump = basic_block.get_jump()
        if next_jump:
            jump_targets.append(cfg_bytecode.get_block_index(next_jump))
        next_block = basic_block.next_block
        if next_block:
            jump_targets.append(cfg_bytecode.get_block_index(next_block))
        for target in jump_targets:
            G.add_edge(curr_index, target)
    return G

def compute_stack_size(cfg: nx.DiGraph):
    sizes = {0:0}
    visited = set()
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        curr_start= sizes[node]
        bb : BasicBlock = cfg.nodes[node]["block"]
        for ins in bb:
            curr_start += ins.stack_effect()
        
        for neighbor in cfg.successors(node):
            sizes[neighbor] = curr_start
            dfs(neighbor)
    dfs(0)
    return sizes


def reverse_postorder(graph):
    """
    Compute the reverse postorder numbering for every node in a directed graph.

    Args:
        graph (nx.DiGraph): A directed graph with natural number nodes.

    Returns:
        dict: A dictionary mapping each node to its reverse postorder number.
    """
    # Ensure the graph is a Directed Graph
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Input graph must be a directed graph (DiGraph).")

    # Perform a depth-first search (DFS) starting from node 0
    postorder = []
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.successors(node):
            dfs(neighbor)
        postorder.append(node)

    dfs(0)  # Assuming 0 is the start node
    # print(postorder)
    # Reverse the postorder to get the numbering
    postorder.reverse()
    reverse_postorder_map = {node: i for i, node in enumerate(postorder)}

    return reverse_postorder_map




@dataclass
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children: list[TreeNode] = []


def build_tree_from_dict(tree_dict):
    """
    Converts a dictionary representing a tree into a tree structure.

    Args:
        tree_dict (dict): A dictionary where keys are nodes and values are parent nodes.

    Returns:
        TreeNode: The root of the tree.
    """
    # Create a mapping of node value to TreeNode instance
    node_map = {node: TreeNode(node) for node in tree_dict.keys()}

    # Add a root node if not explicitly included in the tree_dict
    if 0 not in tree_dict.values():
        root = TreeNode(0)
    else:
        root = node_map[next(node for node, parent in tree_dict.items() if parent == 0)]

    # Construct the tree
    for node, parent in tree_dict.items():
        if node != 0:  # 0 is treated as the root
            node_map[parent].children.append(node_map[node])

    return root


def print_tree(root, level=0):
    """Utility function to print the tree structure."""
    if not root:
        return
    print("  " * level + f"Node({root.val})")
    for child in root.children:
        print_tree(child, level + 1)



@dataclass
class WasmWrapper:
    basic_block: BasicBlock
    
    def __repr__(self) -> str:
        return f"WrapperFor({cfg_bytecode.get_block_index(self.basic_block)})"

@dataclass(kw_only=True)
class WasmStructure:
    in_count : int
    out_count : int 

@dataclass(kw_only=True)
class WasmBlock(WasmStructure):
    blocks: list["Block"]


@dataclass(kw_only=True)
class WasmIf(WasmStructure):
    br1: list["Block"]
    br2: list["Block"]


@dataclass(kw_only=True)
class WasmLoop(WasmStructure):
    blocks: list["Block"]


@dataclass
class WasmBranch:
    target: int


@dataclass
class WasmReturn:
    pass

@dataclass
class WasmUnreachable:
    pass



Block = WasmWrapper | WasmBlock | WasmIf | WasmLoop | WasmBranch | WasmUnreachable


@dataclass
class IfThenElse:
    pass


@dataclass
class LoopHeadedBy:
    label: int


@dataclass
class BlockFolloedBy:
    label: int


ContainingSyntax = IfThenElse | LoopHeadedBy | BlockFolloedBy

Context = list[ContainingSyntax]




@dataclass
class TranslationInfo:
    cfg: nx.DiGraph
    reverse_order_map: dict[any, int]
    full_dom_tree: TreeNode
    stack_sizes : dict[any, int]



def node_with_in(
    info: TranslationInfo,
    node,
    sorted_merge_children: list[TreeNode],
    context: Context,
):
    if len(sorted_merge_children) == 0:
        bb: BasicBlock = info.cfg.nodes[node]["block"]
        content = WasmWrapper(bb)
        successors = list(info.cfg.successors(node))
        assert len(successors) <= 2

        if len(successors) == 0:
            next = [WasmReturn()]
        elif len(successors) == 1:
            next = do_branch(info, node, successors[0], context)
        elif len(successors) == 2:
            new_context = [IfThenElse()] + context
            if info.stack_sizes[successors[0]] != info.stack_sizes[successors[1]]:
                raise RuntimeError("failed branch agreement")
            
            #note that if both branches of the if eventually will branch out, then there shouldn't actually be code following the "following fall behaviour of the if", like there are no code immediately after the if statement since everything should be branched out. In this case, we can say the output type is always nothing
            next = [
                WasmIf(
                    br1=do_branch(info, node, successors[0], new_context),
                    br2=do_branch(info, node, successors[1], new_context),
                    in_count= info.stack_sizes[successors[0]],
                    out_count= 0
                ),
                WasmUnreachable() #this is to ensure that the return type calculation in the validation algorithm correctly recognizes that there is in effect no way for the if to fall off naturally, and therefore will not complain about the stack size unification
            ]
        else:
            raise RuntimeError("more than two branches")
        return [content] + next
    else:
        merge_child = sorted_merge_children[0]
        other_children = sorted_merge_children[1:]
        return [
            WasmBlock(
                blocks=node_with_in(
                    info,
                    node,
                    other_children,
                    [BlockFolloedBy(merge_child.val)] + context,
                ),
                in_count = info.stack_sizes[node],
                out_count= info.stack_sizes[merge_child.val],
            )
        ] + do_tree(info, merge_child, context)


def is_backward(info: TranslationInfo, from_node, to_node):
    return info.reverse_order_map[to_node] <= info.reverse_order_map[from_node]


def is_merge_node(info: TranslationInfo, node):
    """
    N.B. A block is a merge node if it is where control flow merges.
    That means it is entered by multiple control-flow edges, _except_
    back edges don't count.  There must be multiple paths that enter the
    block _without_ passing through the block itself.
    """
    preds = info.cfg.predecessors(node)
    return (
        len( list(
            filter(
                lambda pred: not is_backward(info,pred, node), preds
            ))
        )
        > 1
    )


def is_loop_header(info: TranslationInfo, node):
    preds = info.cfg.predecessors(node)
    return any(
        filter(lambda pred: is_backward(info,pred, node), preds)
    )


def do_tree(info: TranslationInfo, node_dom: TreeNode, context: Context):
    node = node_dom.val

    code_for_node = lambda context: node_with_in(
        info,
        node,
        list(
            sorted(
                filter(
                    lambda tree_child: is_merge_node(info, tree_child.val),
                    node_dom.children,
                ),
                key=lambda tree_node: info.reverse_order_map[tree_node.val],
                reverse=True,
            )
        ),
        context,
    )
    if is_loop_header(info, node):
        
        # similar to the discussion about the if earlier, I don't believe the loop will fall through(in fact the author misunderstood the semantics of the loop)
        return [WasmLoop(blocks=code_for_node([LoopHeadedBy(node)] + context),
                         in_count=info.stack_sizes[node],
                         out_count=0)]
    else:
        return code_for_node(context)


def subtree_at(info: TranslationInfo, target):
    def subtree_at_inner(tree: TreeNode):
        if tree is None:
            return False
        if tree.val == target:
            return tree
        for child in tree.children:
            if subtree := subtree_at_inner(child):
                return subtree

    subtree = subtree_at_inner(info.full_dom_tree)
    if not subtree:
        raise RuntimeError("subtree not found")
    return subtree


def do_branch(info: TranslationInfo, source, target, context: Context):
    get_i = lambda: index(target, context)
    if is_backward(info, source, target) or is_merge_node(info, target):
        return [WasmBranch(get_i())]
    else:
        return do_tree(info, subtree_at(info, target), context)


def index(find_label, context: Context):
    match context:
        case [BlockFolloedBy(label=label), *more_context] if label == find_label:
            return 0
        case [LoopHeadedBy(label=label), *more_context] if label == find_label:
            return 0
        case [_, *more_context]:
            return 1 + index(find_label, more_context)
        case _ :
            raise RuntimeError("destination label not in evaluation context")




# Define labels
L1 = Label()
L2 = Label()
L3 = Label()



# # Construct the bytecode
# code = Bytecode(
#     [
#         Instr("RESUME", 0),
#         Instr("LOAD_FAST", "a"),
#         Instr("LOAD_CONST", 0),
#         Instr("COMPARE_OP", Compare.LT_CAST),
#         Instr("POP_JUMP_IF_FALSE", L1),
#         # If a < 0
#         Instr("LOAD_FAST", "b"),
#         Instr("LOAD_CONST", 3),
#         Instr("BINARY_OP", BinaryOp.MULTIPLY),
#         Instr("LOAD_FAST", "a"),
#         Instr("BINARY_OP", BinaryOp.ADD),
#         Instr("RETURN_VALUE"),
#         # Label L1
#         L1,
#         Instr("LOAD_FAST", "a"),
#         Instr("LOAD_CONST", 0),
#         Instr("COMPARE_OP", Compare.GT_CAST),
#         Instr("POP_JUMP_IF_FALSE", L3),
#         # Label L2 (start of the loop)
#         L2,
#         Instr("LOAD_FAST", "b"),
#         Instr("LOAD_FAST", "a"),
#         Instr("BINARY_OP", BinaryOp.ADD),
#         Instr("STORE_FAST", "b"),
#         Instr("LOAD_FAST", "a"),
#         Instr("LOAD_CONST", 1),
#         Instr("BINARY_OP", BinaryOp.SUBTRACT),
#         Instr("STORE_FAST", "a"),
#         Instr("LOAD_FAST", "a"),
#         Instr("LOAD_CONST", 0),
#         Instr("COMPARE_OP", Compare.GT_CAST),
#         Instr("POP_JUMP_IF_FALSE", L3),
#         Instr("JUMP_BACKWARD", L2),
#         # Label L3
#         L3,
#         Instr("LOAD_CONST", "hello "),
#         Instr("LOAD_FAST", "b"),
#         Instr("FORMAT_SIMPLE"),
#         Instr("BUILD_STRING", 2),
#         Instr("RETURN_VALUE"),
#     ]
# )

code = Bytecode(
    [
        Instr("RESUME", 0),
        
        Instr("LOAD_CONST", 4),
        Instr("LOAD_CONST", 3), 
        Instr("COPY", 1),
        Instr("LOAD_CONST", 10), 
        Instr("COMPARE_OP", Compare.LT_CAST), 
        Instr("POP_JUMP_IF_FALSE", L2),

        # Label L1
        L1,
        Instr("LOAD_CONST", 1),  # 1
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("COPY", 1),
        Instr("LOAD_CONST", 10), 
        Instr("COMPARE_OP", Compare.LT_CAST),  # bool(<)
        Instr("POP_JUMP_IF_FALSE", L2),
        Instr("JUMP_BACKWARD", L1),
        
        # Label L2
        L2,
        Instr("BINARY_OP", BinaryOp.ADD),
        Instr("RETURN_VALUE"),
    ]
)

cfg_bytecode = ControlFlowGraph.from_bytecode(code)
dump_bytecode(cfg_bytecode)
G = cfg_to_network(cfg_bytecode)
imm_dom_tree = build_tree_from_dict(nx.immediate_dominators(G, 0))

reverse_order_map_computed = reverse_postorder(G)
stack_sizes = compute_stack_size(G)
info = TranslationInfo(G, reverse_order_map_computed, imm_dom_tree, stack_sizes)
print(do_tree(info, imm_dom_tree,[]))

print()
# nx.draw(G,  with_labels = True)
# plt.show()

