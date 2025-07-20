"""AST-based dependency analysis for code blocks."""

import builtins
from typing import Optional, Set, Tuple

import libcst

from .module_resolver import is_stdlib_module, resolve_module_to_file

# Python builtin names to exclude from dependencies
BUILTIN_NAMES = set(dir(builtins))


class DependencyVisitor(libcst.CSTVisitor):
    """Extract provides and requires sets from a code block using AST analysis."""

    def __init__(
        self, current_file: Optional[str] = None, project_root: Optional[str] = None
    ):
        self.provides: Set[str] = set()
        self.requires: Set[str] = set()
        self.file_dependencies: Set[str] = set()  # Track file dependencies
        self.scope_stack: list[Set[str]] = [set()]  # Track nested scopes
        self.function_depth = 0
        self.class_depth = 0
        self.current_file = current_file
        self.project_root = project_root

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> Optional[bool]:
        """Function definitions provide their name."""
        if (
            self.function_depth == 0 and self.class_depth == 0
        ):  # Only top-level functions
            self.provides.add(node.name.value)
        elif self.class_depth > 0 and self.function_depth == 0:
            # Method inside a class - add to class scope, not provides
            self.scope_stack[-1].add(node.name.value)

        # Track that we're inside a function
        self.function_depth += 1
        self.scope_stack.append(set())  # New scope for function

        # Add parameters to local scope
        for param in node.params.params:
            if hasattr(param.name, "value"):
                self.scope_stack[-1].add(param.name.value)

        return None  # Continue visiting children

    def leave_FunctionDef(self, original_node: libcst.FunctionDef) -> None:
        """Leave function scope."""
        self.scope_stack.pop()
        self.function_depth -= 1

    def visit_ClassDef(self, node: libcst.ClassDef) -> Optional[bool]:
        """Class definitions provide their name."""
        if self.class_depth == 0:  # Only top-level classes
            self.provides.add(node.name.value)

        # Track that we're inside a class
        self.class_depth += 1
        self.scope_stack.append(set())  # New scope for class

        return None  # Continue visiting children

    def leave_ClassDef(self, original_node: libcst.ClassDef) -> None:
        """Leave class scope."""
        self.scope_stack.pop()
        self.class_depth -= 1

    def visit_Import(self, node: libcst.Import) -> Optional[bool]:
        """Imports provide the imported names and track file dependencies."""
        if isinstance(node.names, libcst.ImportStar):
            # Mark that this block has star imports
            self.provides.add("__star_import__")
        else:
            for alias in node.names:
                if isinstance(alias, libcst.ImportAlias):
                    # Extract the module name
                    module_name = self._get_full_name(alias.name)

                    # Track file dependency if we can resolve it
                    if module_name and self.current_file and self.project_root:
                        if not is_stdlib_module(module_name):
                            file_path = resolve_module_to_file(
                                module_name, self.current_file, self.project_root
                            )
                            if file_path:
                                self.file_dependencies.add(file_path)
                            # Debug
                            # print(f"Import {module_name}: resolved to {file_path}")

                    if alias.asname:
                        # import foo as bar -> provides 'bar'
                        asname = alias.asname.name
                        if isinstance(asname, libcst.Name):
                            self.provides.add(asname.value)
                    else:
                        # import foo.bar -> provides 'foo' (first component)
                        name = alias.name
                        if isinstance(name, libcst.Attribute):
                            # Handle dotted imports
                            first_part = self._get_leftmost_name(name)
                            if first_part:
                                self.provides.add(first_part)
                        elif isinstance(name, libcst.Name):
                            self.provides.add(name.value)

        return False  # Don't visit children of import statements

    def visit_ImportFrom(self, node: libcst.ImportFrom) -> Optional[bool]:
        """From imports provide the imported names and track file dependencies."""
        # Extract module name and relative level
        module_name = ""
        relative_level = 0

        if node.module:
            module_name = self._get_full_name(node.module)

        # Count relative import dots
        if node.relative:
            for dot in node.relative:
                relative_level += 1

        # Track file dependency if we can resolve it
        if self.current_file and self.project_root:
            if not module_name or not is_stdlib_module(module_name):
                file_path = resolve_module_to_file(
                    module_name or "",
                    self.current_file,
                    self.project_root,
                    relative_level,
                )
                if file_path:
                    self.file_dependencies.add(file_path)
                # Debug
                # print(f"ImportFrom {module_name} (level={relative_level}): resolved to {file_path}")

        # Handle imported names
        if isinstance(node.names, libcst.ImportStar):
            # Mark that this block has star imports
            self.provides.add("__star_import__")
        else:
            for alias in node.names:
                if isinstance(alias, libcst.ImportAlias):
                    if alias.asname:
                        # from foo import bar as baz -> provides 'baz'
                        asname = alias.asname.name
                        if isinstance(asname, libcst.Name):
                            self.provides.add(asname.value)
                    else:
                        # from foo import bar -> provides 'bar'
                        name = alias.name
                        if isinstance(name, libcst.Name):
                            self.provides.add(name.value)

        return False  # Don't visit children of import statements

    def visit_Assign(self, node: libcst.Assign) -> Optional[bool]:
        """Assignments provide the target names."""
        # First visit the value to collect any requirements
        if node.value:
            node.value.visit(self)

        # Then extract assignment targets
        for target in node.targets:
            self._extract_assignment_targets(target.target)

        return False  # Don't visit children again (we already visited value)

    def visit_AnnAssign(self, node: libcst.AnnAssign) -> Optional[bool]:
        """Annotated assignments provide the target name."""
        if node.target:
            self._extract_assignment_targets(node.target)

        return None  # Continue visiting children

    def visit_AugAssign(self, node: libcst.AugAssign) -> Optional[bool]:
        """Augmented assignments (+=, etc) require the target to exist."""
        # The target must already exist, so it's a requirement
        if isinstance(node.target, libcst.Name):
            name = node.target.value
            if not self._is_name_in_scope(name):
                self.requires.add(name)

        return None  # Continue visiting children

    def visit_Name(self, node: libcst.Name) -> Optional[bool]:
        """Names that aren't definitions are potential requirements."""
        name = node.value

        # Skip if it's provided by this block
        if name in self.provides:
            return None

        # Skip if it's in any local scope
        for scope in self.scope_stack:
            if name in scope:
                return None

        # Skip builtins
        if name in BUILTIN_NAMES:
            return None

        # This is a free variable - it's required from outside
        self.requires.add(name)

        return None  # Continue visiting children

    def visit_ListComp(self, node: libcst.ListComp) -> Optional[bool]:
        """Handle list comprehension with its own scope."""
        # Create new scope for comprehension
        self.scope_stack.append(set())

        # Add comprehension variables to local scope
        comp_for = node.for_in
        while comp_for:
            self._extract_comp_targets(comp_for.target)
            # Handle nested comprehensions
            comp_for = comp_for.inner_for_in

        return None  # Continue visiting children

    def leave_ListComp(self, original_node: libcst.ListComp) -> None:
        """Leave list comprehension scope."""
        self.scope_stack.pop()

    def visit_SetComp(self, node: libcst.SetComp) -> Optional[bool]:
        """Handle set comprehension with its own scope."""
        # Create new scope for comprehension
        self.scope_stack.append(set())

        # Add comprehension variables to local scope
        comp_for = node.for_in
        while comp_for:
            self._extract_comp_targets(comp_for.target)
            # Handle nested comprehensions
            comp_for = comp_for.inner_for_in

        return None  # Continue visiting children

    def leave_SetComp(self, original_node: libcst.SetComp) -> None:
        """Leave set comprehension scope."""
        self.scope_stack.pop()

    def visit_DictComp(self, node: libcst.DictComp) -> Optional[bool]:
        """Handle dict comprehension with its own scope."""
        # Create new scope for comprehension
        self.scope_stack.append(set())

        # Add comprehension variables to local scope
        comp_for = node.for_in
        while comp_for:
            self._extract_comp_targets(comp_for.target)
            # Handle nested comprehensions
            comp_for = comp_for.inner_for_in

        return None  # Continue visiting children

    def leave_DictComp(self, original_node: libcst.DictComp) -> None:
        """Leave dict comprehension scope."""
        self.scope_stack.pop()

    def visit_GeneratorExp(self, node: libcst.GeneratorExp) -> Optional[bool]:
        """Handle generator expression with its own scope."""
        # Create new scope for comprehension
        self.scope_stack.append(set())

        # Add comprehension variables to local scope
        comp_for = node.for_in
        while comp_for:
            self._extract_comp_targets(comp_for.target)
            # Handle nested comprehensions
            comp_for = comp_for.inner_for_in

        return None  # Continue visiting children

    def leave_GeneratorExp(self, original_node: libcst.GeneratorExp) -> None:
        """Leave generator expression scope."""
        self.scope_stack.pop()

    def _extract_comp_targets(self, target) -> None:
        """Extract comprehension loop variables and add to current scope."""
        if isinstance(target, libcst.Name):
            self.scope_stack[-1].add(target.value)
        elif isinstance(target, (libcst.Tuple, libcst.List)):
            # Handle tuple unpacking in comprehensions
            for element in target.elements:
                if isinstance(element, libcst.Element):
                    self._extract_comp_targets(element.value)

    def visit_Lambda(self, node: libcst.Lambda) -> Optional[bool]:
        """Handle lambda with its own scope."""
        # Create new scope for lambda
        self.scope_stack.append(set())

        # Add lambda parameters to local scope
        for param in node.params.params:
            if hasattr(param.name, "value"):
                self.scope_stack[-1].add(param.name.value)

        return None  # Continue visiting children

    def leave_Lambda(self, original_node: libcst.Lambda) -> None:
        """Leave lambda scope."""
        self.scope_stack.pop()

    def visit_With(self, node: libcst.With) -> Optional[bool]:
        """Handle with statement - targets are provided to outer scope."""
        for item in node.items:
            if item.asname:
                # with ... as name - add to provides
                if isinstance(item.asname.name, libcst.Name):
                    self.provides.add(item.asname.name.value)
                elif isinstance(item.asname.name, (libcst.Tuple, libcst.List)):
                    # Handle tuple unpacking in with statements
                    self._extract_assignment_targets(item.asname.name)

        return None  # Continue visiting children

    def visit_Try(self, node: libcst.Try) -> Optional[bool]:
        """Handle try/except - exception variables are scoped to their handler."""
        # Visit the try body normally
        node.body.visit(self)

        # Handle each except handler
        for handler in node.handlers:
            if isinstance(handler, libcst.ExceptHandler):
                # Visit the exception type (e.g., ValueError, AnalysisError)
                if handler.type:
                    handler.type.visit(self)

                # Create a new scope for the except block
                self.scope_stack.append(set())

                # Add exception variable to the except block's scope (not provides!)
                if handler.name:
                    if isinstance(handler.name, libcst.AsName):
                        name = handler.name.name
                        if isinstance(name, libcst.Name):
                            self.scope_stack[-1].add(name.value)

                # Visit the except body
                handler.body.visit(self)

                # Leave the except scope
                self.scope_stack.pop()

        # Visit orelse and finalbody if they exist
        if node.orelse:
            node.orelse.visit(self)
        if node.finalbody:
            node.finalbody.visit(self)

        return False  # Don't visit children automatically

    def visit_Call(self, node: libcst.Call) -> Optional[bool]:
        """Handle function calls - need to handle keyword arguments specially."""
        # Visit the function being called
        node.func.visit(self)

        # Visit arguments, but skip keyword names
        for arg in node.args:
            # Only visit the value, not the keyword name
            arg.value.visit(self)

        return False  # Don't visit children automatically

    def visit_Attribute(self, node: libcst.Attribute) -> Optional[bool]:
        """Handle attribute access like np.array or module.function."""
        # We only care about the base name for dependencies
        leftmost = self._get_leftmost_name(node)
        if leftmost and leftmost not in self.provides:
            # Check if it's in scope or builtin
            if not self._is_name_in_scope(leftmost) and leftmost not in BUILTIN_NAMES:
                self.requires.add(leftmost)

        # Don't visit the attribute name itself, only the base
        if hasattr(node.value, "visit"):
            node.value.visit(self)
        return False  # Don't visit children automatically

    def _extract_assignment_targets(self, target) -> None:
        """Extract all names from an assignment target."""
        if isinstance(target, libcst.Name):
            # Simple assignment: x = ...
            self.provides.add(target.value)
        elif isinstance(target, libcst.Tuple) or isinstance(target, libcst.List):
            # Tuple/list unpacking: x, y = ... or [x, y] = ...
            for element in target.elements:
                if isinstance(element, libcst.Element):
                    self._extract_assignment_targets(element.value)
                elif isinstance(element, libcst.StarredElement):
                    # Direct StarredElement in tuple (e.g., x, *rest = ...)
                    self._extract_assignment_targets(element)
        elif isinstance(target, libcst.StarredElement):
            # Starred unpacking: *x = ...
            if hasattr(target.value, "value") and isinstance(target.value, libcst.Name):
                self.provides.add(target.value.value)
        # Note: We don't track attribute assignments (obj.x = ...) or subscripts (obj[x] = ...)
        # as they don't create new names in the namespace

    def _get_leftmost_name(self, node) -> Optional[str]:
        """Get the leftmost name from an attribute chain."""
        current = node
        while isinstance(current, libcst.Attribute):
            current = current.value
        if isinstance(current, libcst.Name):
            return current.value
        return None

    def _get_full_name(self, node) -> Optional[str]:
        """Get the full dotted name from a node (e.g., 'package.module')."""
        if isinstance(node, libcst.Name):
            return node.value
        elif isinstance(node, libcst.Attribute):
            parts = []
            current = node
            while isinstance(current, libcst.Attribute):
                if isinstance(current.attr, libcst.Name):
                    parts.append(current.attr.value)
                current = current.value
            if isinstance(current, libcst.Name):
                parts.append(current.value)
            return ".".join(reversed(parts))
        return None

    def _is_name_in_scope(self, name: str) -> bool:
        """Check if a name is in any current scope."""
        for scope in self.scope_stack:
            if name in scope:
                return True
        return False

    def analyze(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Analyze code and return (provides, requires) sets.

        Args:
            code: Python source code to analyze

        Returns:
            Tuple of (provides, requires) where:
            - provides: Set of names this code defines/imports
            - requires: Set of names this code uses but doesn't define
        """
        try:
            module = libcst.parse_module(code)
            # Visit the module to collect provides/requires
            module.visit(self)

            # Clean up requires - remove anything that's also provided
            # (in case of forward references within the block)
            self.requires = self.requires - self.provides

            return self.provides, self.requires
        except Exception:
            # Parse errors shouldn't crash analysis
            # Return empty sets to indicate no dependencies could be determined
            return set(), set()


def analyze_block(
    code: str, current_file: Optional[str] = None, project_root: Optional[str] = None
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Analyze a code block's dependencies including file dependencies.

    Args:
        code: Python source code
        current_file: Path to the current file (relative to project root)
        project_root: Path to the project root directory

    Returns:
        Tuple of (provides, requires, file_dependencies) sets
    """
    visitor = DependencyVisitor(current_file, project_root)
    provides, requires = visitor.analyze(code)
    return provides, requires, visitor.file_dependencies
