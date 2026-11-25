import hashlib
from typing import Optional

import tree_sitter_c as tsc
from tree_sitter import Language, Node, Parser

_LANGUAGE: Language = Language(tsc.language())
_PARSER: Parser = Parser(_LANGUAGE)


def _find_identifier_in_declarator(node: Node) -> Optional[str]:
    if node.type == "identifier":
        return node.text.decode("utf-8")

    for child in node.children:
        result = _find_identifier_in_declarator(child)
        if result is not None:
            return result

    return None


def _extract_function_name(func_node: Node) -> Optional[str]:
    for child in func_node.children:
        if child.type in ("function_declarator", "pointer_declarator"):
            return _find_identifier_in_declarator(child)
    return None


def _collect_function_nodes(node: Node, results: list) -> None:
    if node.type == "function_definition":
        results.append(node)
        return
    for child in node.children:
        _collect_function_nodes(child, results)


def extract_functions_from_code(code_content: str) -> dict[str, str]:
    source_bytes: bytes = code_content.encode("utf-8", errors="replace")
    tree = _PARSER.parse(source_bytes)

    func_nodes: list[Node] = []
    _collect_function_nodes(tree.root_node, func_nodes)

    functions: dict[str, str] = {}
    for func_node in func_nodes:
        name = _extract_function_name(func_node)
        if name is None:
            continue
        func_text: str = func_node.text.decode("utf-8", errors="replace")
        functions[name] = func_text

    return functions


def canonical(node: Node, source: bytes) -> str:
    if node.is_extra and "comment" in node.type:
        return ""
    if node.child_count == 0:
        text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        return f"{node.type}:{text}"

    parts = [canonical(c, source) for c in node.children]
    parts = [p for p in parts if p]
    joined = " ".join(parts)
    return f"({node.type} {joined})"


def calculate_ast_hash(code_content: str) -> str:
    source_bytes: bytes = code_content.encode("utf-8", errors="replace")
    tree = _PARSER.parse(source_bytes)
    sig = canonical(tree.root_node, source_bytes)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()
