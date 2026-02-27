"""
Code Analyzer for Security Evaluation.

Provides language-aware code analysis including:
- AST parsing (Python native, regex-based for others)
- Function/class extraction
- String literal extraction
- Import analysis
- Language detection
"""

import ast
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about a function definition."""
    name: str
    line: int
    end_line: Optional[int]
    args: List[str]
    decorators: List[str]
    is_async: bool
    docstring: Optional[str]
    calls: List[str]  # Functions called within this function


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]  # Specific names imported (empty for "import module")
    alias: Optional[str]
    line: int


@dataclass
class StringLiteral:
    """A string literal found in code."""
    value: str
    line: int
    column: Optional[int]
    is_fstring: bool
    context: Optional[str]  # e.g., "sql_query", "file_path", "command"


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    language: str
    functions: List[FunctionInfo]
    imports: List[ImportInfo]
    strings: List[StringLiteral]
    variables: Dict[str, List[int]]  # Variable name -> lines where assigned
    dangerous_calls: List[Tuple[str, int]]  # (function_name, line)
    parse_errors: List[str]
    raw_ast: Optional[Any]  # Language-specific AST


class LanguageAnalyzer(ABC):
    """Base class for language-specific code analyzers."""

    language: str = "unknown"

    # Dangerous functions to track
    DANGEROUS_FUNCTIONS: List[str] = []

    @abstractmethod
    def parse(self, code: str) -> Optional[Any]:
        """Parse code into AST or structured representation."""
        pass

    @abstractmethod
    def analyze(self, code: str) -> AnalysisResult:
        """Perform full analysis of code."""
        pass

    def _find_dangerous_calls(
        self, code: str, functions: Optional[List[str]] = None
    ) -> List[Tuple[str, int]]:
        """Find calls to dangerous functions."""
        dangerous = functions or self.DANGEROUS_FUNCTIONS
        calls = []

        for i, line in enumerate(code.split("\n"), 1):
            for func in dangerous:
                # Handle both module.function and function patterns
                patterns = [
                    rf"\b{re.escape(func)}\s*\(",
                    rf"\b{func.split('.')[-1]}\s*\(" if "." in func else None,
                ]
                for pattern in patterns:
                    if pattern and re.search(pattern, line):
                        calls.append((func, i))
                        break

        return calls


class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer using the ast module."""

    language = "python"

    DANGEROUS_FUNCTIONS = [
        # Command execution
        "os.system", "os.popen", "os.spawn", "os.spawnl", "os.spawnle",
        "subprocess.call", "subprocess.run", "subprocess.Popen",
        "subprocess.check_output", "subprocess.check_call",
        "commands.getoutput", "commands.getstatusoutput",
        # Code execution
        "eval", "exec", "compile", "execfile",
        # Deserialization
        "pickle.loads", "pickle.load", "cPickle.loads", "cPickle.load",
        "yaml.load", "yaml.unsafe_load",
        "marshal.loads", "marshal.load",
        # SQL
        "cursor.execute", "connection.execute", "db.execute",
        "engine.execute", "session.execute",
        # File operations
        "open", "file",
        # Network
        "urllib.urlopen", "urllib2.urlopen", "requests.get", "requests.post",
        "httplib.HTTPConnection", "http.client.HTTPConnection",
        # Crypto (weak)
        "hashlib.md5", "hashlib.sha1",
        "Crypto.Cipher.DES", "Crypto.Cipher.ARC4",
        # Random (insecure)
        "random.random", "random.randint", "random.choice",
    ]

    def parse(self, code: str) -> Optional[ast.AST]:
        """Parse Python code into AST."""
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze Python code."""
        tree = self.parse(code)
        parse_errors = []

        if tree is None:
            # Try to extract what we can with regex
            parse_errors.append("Failed to parse Python AST")
            return AnalysisResult(
                language=self.language,
                functions=self._extract_functions_regex(code),
                imports=self._extract_imports_regex(code),
                strings=self._extract_strings_regex(code),
                variables={},
                dangerous_calls=self._find_dangerous_calls(code),
                parse_errors=parse_errors,
                raw_ast=None,
            )

        return AnalysisResult(
            language=self.language,
            functions=self._extract_functions(tree),
            imports=self._extract_imports(tree),
            strings=self._extract_strings(tree, code),
            variables=self._extract_variables(tree),
            dangerous_calls=self._find_dangerous_calls_ast(tree),
            parse_errors=parse_errors,
            raw_ast=tree,
        )

    def _extract_functions(self, tree: ast.AST) -> List[FunctionInfo]:
        """Extract function definitions from AST."""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get decorator names
                decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorators.append(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorators.append(f"{self._get_full_name(dec)}")
                    elif isinstance(dec, ast.Call):
                        if isinstance(dec.func, ast.Name):
                            decorators.append(dec.func.id)
                        elif isinstance(dec.func, ast.Attribute):
                            decorators.append(self._get_full_name(dec.func))

                # Get docstring
                docstring = ast.get_docstring(node)

                # Get function calls within this function
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = self._get_call_name(child)
                        if call_name:
                            calls.append(call_name)

                functions.append(FunctionInfo(
                    name=node.name,
                    line=node.lineno,
                    end_line=getattr(node, 'end_lineno', None),
                    args=[arg.arg for arg in node.args.args],
                    decorators=decorators,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    docstring=docstring,
                    calls=calls,
                ))

        return functions

    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        line=node.lineno,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    alias=None,
                    line=node.lineno,
                ))

        return imports

    def _extract_strings(self, tree: ast.AST, code: str) -> List[StringLiteral]:
        """Extract string literals from AST."""
        strings = []
        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Determine context
                context = self._determine_string_context(node, tree, lines)
                strings.append(StringLiteral(
                    value=node.value,
                    line=node.lineno,
                    column=getattr(node, 'col_offset', None),
                    is_fstring=False,
                    context=context,
                ))
            elif isinstance(node, ast.JoinedStr):  # f-string
                # Reconstruct f-string value
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                    elif isinstance(value, ast.FormattedValue):
                        parts.append("{...}")

                strings.append(StringLiteral(
                    value="".join(parts),
                    line=node.lineno,
                    column=getattr(node, 'col_offset', None),
                    is_fstring=True,
                    context=self._determine_string_context(node, tree, lines),
                ))

        return strings

    def _determine_string_context(
        self, node: ast.AST, tree: ast.AST, lines: List[str]
    ) -> Optional[str]:
        """Determine the context of a string (SQL, command, path, etc.)."""
        line_content = lines[node.lineno - 1].lower() if node.lineno <= len(lines) else ""

        # Check for SQL keywords
        sql_keywords = ["select", "insert", "update", "delete", "drop", "create", "alter"]
        if any(kw in line_content for kw in sql_keywords):
            return "sql_query"

        # Check for file path indicators
        if any(p in line_content for p in ["open(", "file(", "path", "filename"]):
            return "file_path"

        # Check for command execution
        if any(c in line_content for c in ["os.system", "subprocess", "popen", "exec"]):
            return "command"

        # Check for URL
        if any(u in line_content for u in ["http://", "https://", "url", "requests"]):
            return "url"

        return None

    def _extract_variables(self, tree: ast.AST) -> Dict[str, List[int]]:
        """Extract variable assignments."""
        variables: Dict[str, List[int]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in variables:
                            variables[target.id] = []
                        variables[target.id].append(node.lineno)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if name not in variables:
                    variables[name] = []
                variables[name].append(node.lineno)

        return variables

    def _find_dangerous_calls_ast(self, tree: ast.AST) -> List[Tuple[str, int]]:
        """Find dangerous function calls using AST."""
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    # Check if it's in our dangerous list
                    for dangerous in self.DANGEROUS_FUNCTIONS:
                        if call_name == dangerous or call_name.endswith(f".{dangerous}"):
                            calls.append((call_name, node.lineno))
                            break
                        # Also check if the last part matches
                        if "." in dangerous:
                            _, func_name = dangerous.rsplit(".", 1)
                            if call_name.endswith(func_name):
                                calls.append((call_name, node.lineno))
                                break

        return calls

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_full_name(node.func)
        return None

    def _get_full_name(self, node: ast.Attribute) -> str:
        """Get full dotted name from Attribute node."""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))

    def _extract_functions_regex(self, code: str) -> List[FunctionInfo]:
        """Fallback regex-based function extraction."""
        functions = []
        pattern = r"^\s*(async\s+)?def\s+(\w+)\s*\((.*?)\)"

        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(pattern, line)
            if match:
                is_async = bool(match.group(1))
                name = match.group(2)
                args_str = match.group(3)
                args = [a.strip().split(":")[0].split("=")[0].strip()
                        for a in args_str.split(",") if a.strip()]

                functions.append(FunctionInfo(
                    name=name,
                    line=i,
                    end_line=None,
                    args=args,
                    decorators=[],
                    is_async=is_async,
                    docstring=None,
                    calls=[],
                ))

        return functions

    def _extract_imports_regex(self, code: str) -> List[ImportInfo]:
        """Fallback regex-based import extraction."""
        imports = []

        # import module
        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", line)
            if match:
                imports.append(ImportInfo(
                    module=match.group(1),
                    names=[],
                    alias=match.group(2),
                    line=i,
                ))

            # from module import names
            match = re.match(r"^\s*from\s+([\w.]+)\s+import\s+(.+)", line)
            if match:
                names = [n.strip().split(" as ")[0] for n in match.group(2).split(",")]
                imports.append(ImportInfo(
                    module=match.group(1),
                    names=names,
                    alias=None,
                    line=i,
                ))

        return imports

    def _extract_strings_regex(self, code: str) -> List[StringLiteral]:
        """Fallback regex-based string extraction."""
        strings = []

        # Match various string patterns
        patterns = [
            (r'f["\']([^"\']*)["\']', True),   # f-strings
            (r'["\']([^"\']{3,})["\']', False),  # Regular strings (3+ chars)
        ]

        for i, line in enumerate(code.split("\n"), 1):
            for pattern, is_fstring in patterns:
                for match in re.finditer(pattern, line):
                    strings.append(StringLiteral(
                        value=match.group(1),
                        line=i,
                        column=match.start(),
                        is_fstring=is_fstring,
                        context=None,
                    ))

        return strings


class JavaScriptAnalyzer(LanguageAnalyzer):
    """JavaScript/TypeScript analyzer using regex patterns."""

    language = "javascript"

    DANGEROUS_FUNCTIONS = [
        # Code execution
        "eval", "Function", "setTimeout", "setInterval",
        # Command execution
        "exec", "execSync", "spawn", "spawnSync", "execFile",
        "child_process.exec", "child_process.spawn",
        # DOM manipulation (XSS)
        "innerHTML", "outerHTML", "document.write", "document.writeln",
        # SQL (common ORMs)
        "query", "raw", "execute",
        # Deserialization
        "JSON.parse",
    ]

    def parse(self, code: str) -> Optional[Dict[str, Any]]:
        """Parse JavaScript using regex (tree-sitter optional)."""
        # Basic structural analysis without full parsing
        return {"code": code}

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze JavaScript code."""
        return AnalysisResult(
            language=self.language,
            functions=self._extract_functions(code),
            imports=self._extract_imports(code),
            strings=self._extract_strings(code),
            variables=self._extract_variables(code),
            dangerous_calls=self._find_dangerous_calls(code),
            parse_errors=[],
            raw_ast=None,
        )

    def _extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract function definitions."""
        functions = []

        patterns = [
            # function name(args)
            r"function\s+(\w+)\s*\((.*?)\)",
            # const/let/var name = function(args)
            r"(?:const|let|var)\s+(\w+)\s*=\s*function\s*\((.*?)\)",
            # const/let/var name = (args) =>
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\((.*?)\)\s*=>",
            # async function name(args)
            r"async\s+function\s+(\w+)\s*\((.*?)\)",
        ]

        for i, line in enumerate(code.split("\n"), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    args_str = match.group(2)
                    args = [a.strip().split(":")[0].split("=")[0].strip()
                            for a in args_str.split(",") if a.strip()]

                    is_async = "async" in line[:match.start()]

                    functions.append(FunctionInfo(
                        name=name,
                        line=i,
                        end_line=None,
                        args=args,
                        decorators=[],
                        is_async=is_async,
                        docstring=None,
                        calls=[],
                    ))
                    break

        return functions

    def _extract_imports(self, code: str) -> List[ImportInfo]:
        """Extract import statements."""
        imports = []

        patterns = [
            # import x from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import { x, y } from 'module'
            r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]",
            # const x = require('module')
            r"(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(['\"]([^'\"]+)['\"]\)",
        ]

        for i, line in enumerate(code.split("\n"), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    if "{" in pattern:
                        names = [n.strip() for n in match.group(1).split(",")]
                        module = match.group(2)
                    else:
                        names = [match.group(1)]
                        module = match.group(2)

                    imports.append(ImportInfo(
                        module=module,
                        names=names,
                        alias=None,
                        line=i,
                    ))

        return imports

    def _extract_strings(self, code: str) -> List[StringLiteral]:
        """Extract string literals."""
        strings = []

        for i, line in enumerate(code.split("\n"), 1):
            # Template literals
            for match in re.finditer(r"`([^`]*)`", line):
                strings.append(StringLiteral(
                    value=match.group(1),
                    line=i,
                    column=match.start(),
                    is_fstring="${" in match.group(1),
                    context=None,
                ))

            # Regular strings
            for match in re.finditer(r"['\"]([^'\"]{3,})['\"]", line):
                if not re.search(r"['\"]" + re.escape(match.group(1)) + r"['\"]", line[:match.start()]):
                    strings.append(StringLiteral(
                        value=match.group(1),
                        line=i,
                        column=match.start(),
                        is_fstring=False,
                        context=None,
                    ))

        return strings

    def _extract_variables(self, code: str) -> Dict[str, List[int]]:
        """Extract variable assignments."""
        variables: Dict[str, List[int]] = {}

        pattern = r"(?:const|let|var)\s+(\w+)\s*="
        for i, line in enumerate(code.split("\n"), 1):
            for match in re.finditer(pattern, line):
                name = match.group(1)
                if name not in variables:
                    variables[name] = []
                variables[name].append(i)

        return variables


class JavaAnalyzer(LanguageAnalyzer):
    """Java analyzer using regex patterns."""

    language = "java"

    DANGEROUS_FUNCTIONS = [
        # Command execution
        "Runtime.exec", "ProcessBuilder",
        # SQL
        "executeQuery", "executeUpdate", "prepareStatement",
        "createStatement", "Statement.execute",
        # Deserialization
        "ObjectInputStream.readObject", "XMLDecoder.readObject",
        # File operations
        "FileInputStream", "FileOutputStream", "FileReader", "FileWriter",
        # Crypto (weak)
        "MessageDigest.getInstance",
    ]

    def parse(self, code: str) -> Optional[Dict[str, Any]]:
        """Parse Java using regex."""
        return {"code": code}

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze Java code."""
        return AnalysisResult(
            language=self.language,
            functions=self._extract_functions(code),
            imports=self._extract_imports(code),
            strings=self._extract_strings(code),
            variables=self._extract_variables(code),
            dangerous_calls=self._find_dangerous_calls(code),
            parse_errors=[],
            raw_ast=None,
        )

    def _extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract method definitions."""
        functions = []

        pattern = r"(?:public|private|protected)?\s*(?:static)?\s*(?:\w+)\s+(\w+)\s*\((.*?)\)"

        for i, line in enumerate(code.split("\n"), 1):
            match = re.search(pattern, line)
            if match and match.group(1) not in ["if", "while", "for", "switch"]:
                name = match.group(1)
                args_str = match.group(2)
                args = []
                if args_str.strip():
                    for arg in args_str.split(","):
                        parts = arg.strip().split()
                        if len(parts) >= 2:
                            args.append(parts[-1])

                functions.append(FunctionInfo(
                    name=name,
                    line=i,
                    end_line=None,
                    args=args,
                    decorators=[],
                    is_async=False,
                    docstring=None,
                    calls=[],
                ))

        return functions

    def _extract_imports(self, code: str) -> List[ImportInfo]:
        """Extract import statements."""
        imports = []

        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(r"import\s+([\w.]+);", line.strip())
            if match:
                imports.append(ImportInfo(
                    module=match.group(1),
                    names=[],
                    alias=None,
                    line=i,
                ))

        return imports

    def _extract_strings(self, code: str) -> List[StringLiteral]:
        """Extract string literals."""
        strings = []

        for i, line in enumerate(code.split("\n"), 1):
            for match in re.finditer(r'"([^"]{3,})"', line):
                strings.append(StringLiteral(
                    value=match.group(1),
                    line=i,
                    column=match.start(),
                    is_fstring=False,
                    context=None,
                ))

        return strings

    def _extract_variables(self, code: str) -> Dict[str, List[int]]:
        """Extract variable declarations."""
        variables: Dict[str, List[int]] = {}

        pattern = r"(?:final\s+)?(?:\w+)\s+(\w+)\s*="
        for i, line in enumerate(code.split("\n"), 1):
            for match in re.finditer(pattern, line):
                name = match.group(1)
                if name not in variables:
                    variables[name] = []
                variables[name].append(i)

        return variables


class GoAnalyzer(LanguageAnalyzer):
    """Go analyzer using regex patterns."""

    language = "go"

    DANGEROUS_FUNCTIONS = [
        # Command execution
        "exec.Command", "os/exec.Command",
        # SQL
        "db.Query", "db.Exec", "db.QueryRow",
        # File operations
        "os.Open", "os.Create", "ioutil.ReadFile", "ioutil.WriteFile",
        # Crypto (weak)
        "md5.New", "sha1.New",
    ]

    def parse(self, code: str) -> Optional[Dict[str, Any]]:
        """Parse Go using regex."""
        return {"code": code}

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze Go code."""
        return AnalysisResult(
            language=self.language,
            functions=self._extract_functions(code),
            imports=self._extract_imports(code),
            strings=self._extract_strings(code),
            variables=self._extract_variables(code),
            dangerous_calls=self._find_dangerous_calls(code),
            parse_errors=[],
            raw_ast=None,
        )

    def _extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract function definitions."""
        functions = []

        pattern = r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\((.*?)\)"

        for i, line in enumerate(code.split("\n"), 1):
            match = re.search(pattern, line)
            if match:
                name = match.group(1)
                args_str = match.group(2)
                args = []
                if args_str.strip():
                    for arg in args_str.split(","):
                        parts = arg.strip().split()
                        if parts:
                            args.append(parts[0])

                functions.append(FunctionInfo(
                    name=name,
                    line=i,
                    end_line=None,
                    args=args,
                    decorators=[],
                    is_async=False,
                    docstring=None,
                    calls=[],
                ))

        return functions

    def _extract_imports(self, code: str) -> List[ImportInfo]:
        """Extract import statements."""
        imports = []

        # Single import
        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(r'import\s+"([^"]+)"', line.strip())
            if match:
                imports.append(ImportInfo(
                    module=match.group(1),
                    names=[],
                    alias=None,
                    line=i,
                ))

        # Multi-line import block
        in_import = False
        for i, line in enumerate(code.split("\n"), 1):
            if "import (" in line:
                in_import = True
                continue
            if in_import:
                if ")" in line:
                    in_import = False
                    continue
                match = re.search(r'"([^"]+)"', line)
                if match:
                    imports.append(ImportInfo(
                        module=match.group(1),
                        names=[],
                        alias=None,
                        line=i,
                    ))

        return imports

    def _extract_strings(self, code: str) -> List[StringLiteral]:
        """Extract string literals."""
        strings = []

        for i, line in enumerate(code.split("\n"), 1):
            # Double-quoted strings
            for match in re.finditer(r'"([^"]{3,})"', line):
                strings.append(StringLiteral(
                    value=match.group(1),
                    line=i,
                    column=match.start(),
                    is_fstring=False,
                    context=None,
                ))

            # Backtick strings (raw)
            for match in re.finditer(r"`([^`]{3,})`", line):
                strings.append(StringLiteral(
                    value=match.group(1),
                    line=i,
                    column=match.start(),
                    is_fstring=False,
                    context=None,
                ))

        return strings

    def _extract_variables(self, code: str) -> Dict[str, List[int]]:
        """Extract variable declarations."""
        variables: Dict[str, List[int]] = {}

        patterns = [
            r"(\w+)\s*:=",  # Short declaration
            r"var\s+(\w+)",  # Var declaration
        ]

        for i, line in enumerate(code.split("\n"), 1):
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    name = match.group(1)
                    if name not in variables:
                        variables[name] = []
                    variables[name].append(i)

        return variables


class CodeAnalyzer:
    """
    Main code analyzer that delegates to language-specific analyzers.

    Usage:
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code, "python")

        # Auto-detect language
        result = analyzer.analyze(code)
    """

    ANALYZERS: Dict[str, type] = {
        "python": PythonAnalyzer,
        "py": PythonAnalyzer,
        "javascript": JavaScriptAnalyzer,
        "js": JavaScriptAnalyzer,
        "typescript": JavaScriptAnalyzer,
        "ts": JavaScriptAnalyzer,
        "java": JavaAnalyzer,
        "go": GoAnalyzer,
        "golang": GoAnalyzer,
    }

    # Language detection patterns
    LANGUAGE_PATTERNS: Dict[str, List[str]] = {
        "python": [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import",
            r"^\s*class\s+\w+.*:",
            r"^\s*if\s+__name__\s*==",
            r"print\s*\(",
        ],
        "javascript": [
            r"^\s*const\s+\w+\s*=",
            r"^\s*let\s+\w+\s*=",
            r"^\s*function\s+\w+\s*\(",
            r"=>\s*{",
            r"require\s*\(",
            r"module\.exports",
            r"console\.log",
        ],
        "java": [
            r"^\s*public\s+class",
            r"^\s*private\s+\w+",
            r"^\s*import\s+java\.",
            r"System\.out\.println",
            r"public\s+static\s+void\s+main",
        ],
        "go": [
            r"^\s*package\s+\w+",
            r"^\s*func\s+\w+\s*\(",
            r"^\s*import\s+\(",
            r"fmt\.Print",
            r":=",
        ],
    }

    def __init__(self):
        self._analyzers: Dict[str, LanguageAnalyzer] = {}

    def get_analyzer(self, language: str) -> Optional[LanguageAnalyzer]:
        """Get or create an analyzer for the given language."""
        lang_lower = language.lower()
        if lang_lower not in self._analyzers:
            analyzer_cls = self.ANALYZERS.get(lang_lower)
            if analyzer_cls:
                self._analyzers[lang_lower] = analyzer_cls()
        return self._analyzers.get(lang_lower)

    def detect_language(self, code: str) -> str:
        """Auto-detect the programming language of code."""
        scores: Dict[str, int] = {lang: 0 for lang in self.LANGUAGE_PATTERNS}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    scores[lang] += 1

        # Return language with highest score, default to python
        if max(scores.values()) == 0:
            return "python"

        return max(scores, key=lambda k: scores[k])

    def analyze(
        self, code: str, language: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze code for security evaluation.

        Args:
            code: Source code to analyze
            language: Programming language (auto-detected if not provided)

        Returns:
            AnalysisResult with functions, imports, strings, etc.
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(code)

        analyzer = self.get_analyzer(language)
        if analyzer is None:
            # Return basic analysis for unsupported languages
            return AnalysisResult(
                language=language,
                functions=[],
                imports=[],
                strings=[],
                variables={},
                dangerous_calls=[],
                parse_errors=[f"Unsupported language: {language}"],
                raw_ast=None,
            )

        return analyzer.analyze(code)

    def supports_language(self, language: str) -> bool:
        """Check if a language is supported."""
        return language.lower() in self.ANALYZERS

    def supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(set(
            analyzer_cls.language
            for analyzer_cls in self.ANALYZERS.values()
        ))
