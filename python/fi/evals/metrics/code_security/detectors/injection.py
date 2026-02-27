"""
Injection Vulnerability Detectors.

Detects various injection vulnerabilities:
- CWE-89: SQL Injection
- CWE-78: OS Command Injection
- CWE-79: Cross-Site Scripting (XSS)
- CWE-94: Code Injection
- CWE-611: XXE Injection
- CWE-918: SSRF
"""

import re
from typing import List, Optional, Dict, Set

from .base import BaseDetector, PatternBasedDetector, register_detector
from ..types import (
    SecurityFinding,
    Severity,
    VulnerabilityCategory,
)
from ..analyzer import AnalysisResult


@register_detector("sql_injection")
class SQLInjectionDetector(BaseDetector):
    """
    Detects SQL injection vulnerabilities (CWE-89).

    Identifies:
    - String concatenation in SQL queries
    - f-string/format string in SQL
    - Raw string interpolation
    - Missing parameterized queries

    Examples of vulnerable code:
        query = "SELECT * FROM users WHERE id = " + user_id
        query = f"SELECT * FROM users WHERE name = '{name}'"
        cursor.execute("SELECT * FROM users WHERE id = %s" % user_id)
    """

    name = "sql_injection"
    cwe_ids = ["CWE-89"]
    category = VulnerabilityCategory.INJECTION
    description = "SQL Injection via string concatenation or interpolation"
    default_severity = Severity.HIGH

    # SQL keywords to look for
    SQL_KEYWORDS = [
        "select", "insert", "update", "delete", "drop", "create",
        "alter", "exec", "execute", "union", "truncate", "grant",
    ]

    # Patterns indicating unsafe SQL construction
    UNSAFE_PATTERNS = {
        "python": [
            # f-string with SQL
            (r'f["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*\{', "f-string SQL interpolation"),
            # String concatenation
            (r'["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*["\']\s*\+', "String concatenation in SQL"),
            (r'\+\s*["\'].*(?:' + "|".join(SQL_KEYWORDS) + r')', "String concatenation in SQL"),
            # % formatting
            (r'["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*%s.*["\']\s*%', "%-format SQL interpolation"),
            # .format()
            (r'["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*\{\}.*["\']\.format\(', ".format() SQL interpolation"),
        ],
        "javascript": [
            # Template literal
            (r'`.*(?:' + "|".join(SQL_KEYWORDS) + r').*\$\{', "Template literal SQL interpolation"),
            # String concatenation
            (r'["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*["\']\s*\+', "String concatenation in SQL"),
        ],
        "java": [
            # String concatenation
            (r'".*(?:' + "|".join(SQL_KEYWORDS) + r').*"\s*\+', "String concatenation in SQL"),
            # String.format
            (r'String\.format\s*\(\s*".*(?:' + "|".join(SQL_KEYWORDS) + r')', "String.format SQL"),
        ],
        "php": [
            # Variable interpolation
            (r'".*(?:' + "|".join(SQL_KEYWORDS) + r').*\$', "Variable interpolation in SQL"),
            # String concatenation
            (r'["\'].*(?:' + "|".join(SQL_KEYWORDS) + r').*["\']\s*\.', "String concatenation in SQL"),
        ],
    }

    # Safe patterns (parameterized queries)
    SAFE_PATTERNS = {
        "python": [
            r"execute\s*\([^,]+,\s*[\[\(]",  # execute(query, (params,))
            r"executemany\s*\([^,]+,",  # executemany(query, params)
            r"\.filter\s*\(",  # ORM filter
            r"\.where\s*\(",  # ORM where
        ],
        "javascript": [
            r"\?\s*,",  # Prepared statement placeholder
            r"query\s*\([^,]+,\s*\[",  # query(sql, [params])
        ],
        "java": [
            r"prepareStatement",  # Prepared statement
            r"setString\s*\(",  # Parameter binding
            r"setInt\s*\(",
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect SQL injection vulnerabilities."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        # Get patterns for this language
        unsafe_patterns = self.UNSAFE_PATTERNS.get(lang_lower, self.UNSAFE_PATTERNS["python"])
        safe_patterns = self.SAFE_PATTERNS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Skip if no SQL keywords
            if not any(kw in line_lower for kw in self.SQL_KEYWORDS):
                continue

            # Skip if matches safe pattern
            if any(re.search(p, line, re.IGNORECASE) for p in safe_patterns):
                continue

            # Check unsafe patterns
            for pattern, description in unsafe_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Determine confidence based on context
                    confidence = 0.8
                    if analysis and analysis.functions:
                        # Higher confidence if in a route/endpoint
                        func = self._find_function(analysis, i)
                        if func and any(d in ["route", "get", "post", "api"] for d in func.decorators):
                            confidence = 0.95

                    findings.append(self.create_finding(
                        vulnerability_type="SQL Injection",
                        description=f"{description}: {line.strip()[:60]}...",
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=confidence,
                        suggested_fix="Use parameterized queries: cursor.execute(query, (param,))",
                    ))
                    break  # One finding per line

        return findings

    def _find_function(self, analysis: AnalysisResult, line: int):
        """Find the function containing the given line."""
        for func in analysis.functions:
            if func.line <= line and (func.end_line is None or func.end_line >= line):
                return func
        return None


@register_detector("command_injection")
class CommandInjectionDetector(BaseDetector):
    """
    Detects OS command injection vulnerabilities (CWE-78).

    Identifies:
    - os.system() with variable input
    - subprocess with shell=True
    - eval/exec with user input
    - Backtick execution

    Examples of vulnerable code:
        os.system("ping " + host)
        subprocess.call(cmd, shell=True)
        eval(user_input)
    """

    name = "command_injection"
    cwe_ids = ["CWE-78"]
    category = VulnerabilityCategory.INJECTION
    description = "OS Command Injection via shell execution"
    default_severity = Severity.CRITICAL

    DANGEROUS_FUNCTIONS = {
        "python": {
            "os.system": "Shell command execution",
            "os.popen": "Shell command execution",
            "subprocess.call": "Subprocess execution",
            "subprocess.run": "Subprocess execution",
            "subprocess.Popen": "Subprocess execution",
            "subprocess.check_output": "Subprocess execution",
            "subprocess.check_call": "Subprocess execution",
            "commands.getoutput": "Shell command execution",
            "commands.getstatusoutput": "Shell command execution",
            "eval": "Code evaluation",
            "exec": "Code execution",
        },
        "javascript": {
            "eval": "Code evaluation",
            "exec": "Shell command execution",
            "execSync": "Shell command execution",
            "spawn": "Process spawn",
            "spawnSync": "Process spawn",
            "execFile": "File execution",
            "child_process.exec": "Shell command execution",
            "child_process.execSync": "Shell command execution",
        },
        "java": {
            "Runtime.exec": "Shell command execution",
            "Runtime.getRuntime().exec": "Shell command execution",
            "ProcessBuilder": "Process execution",
        },
        "php": {
            "exec": "Shell command execution",
            "shell_exec": "Shell command execution",
            "system": "Shell command execution",
            "passthru": "Shell command execution",
            "popen": "Shell command execution",
            "proc_open": "Process execution",
        },
    }

    # Patterns indicating variable/user input
    VARIABLE_INPUT_PATTERNS = [
        r"\([^)]*\+",  # function(... +
        r"\(f['\"]",   # function(f"...
        r"\([^)]*\{",  # function(...{
        r"\([^)]*\$",  # function(...$
        r"\([^)]*%",   # function(... %
        r"shell\s*=\s*True",  # shell=True
        r"`[^`]*\$\{",  # Backtick with ${...}
    ]

    # Backtick execution patterns (shell command in template literal)
    BACKTICK_PATTERNS = [
        (r"`[^`]*\$\{[^}]+\}[^`]*`", "backtick template with variable"),
    ]

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect command injection vulnerabilities."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        dangerous_funcs = self.DANGEROUS_FUNCTIONS.get(lang_lower, {})

        for i, line in enumerate(lines, 1):
            # Check for backtick command execution (JavaScript)
            if lang_lower in ["javascript", "js", "typescript", "ts"]:
                for pattern, description in self.BACKTICK_PATTERNS:
                    if re.search(pattern, line):
                        findings.append(self.create_finding(
                            vulnerability_type="Command Injection",
                            description=f"Shell command via {description}",
                            line=i,
                            snippet=line.strip()[:100],
                            severity=Severity.CRITICAL,
                            confidence=0.85,
                            suggested_fix="Avoid template literal shell commands. Use proper command builders.",
                        ))
                        break

            for func, description in dangerous_funcs.items():
                # Check if function is called
                func_pattern = rf"\b{re.escape(func.split('.')[-1])}\s*\("
                if not re.search(func_pattern, line):
                    continue

                # Check for variable input or shell=True
                has_variable_input = any(
                    re.search(p, line) for p in self.VARIABLE_INPUT_PATTERNS
                )

                if has_variable_input or "shell=True" in line.lower():
                    severity = Severity.CRITICAL
                    confidence = 0.9 if has_variable_input else 0.85

                    findings.append(self.create_finding(
                        vulnerability_type="Command Injection",
                        description=f"{description} with variable input: {func}",
                        line=i,
                        snippet=line.strip()[:100],
                        severity=severity,
                        confidence=confidence,
                        suggested_fix="Use subprocess with shell=False and list arguments. Validate/sanitize input.",
                    ))

        return findings


@register_detector("xss")
class XSSDetector(BaseDetector):
    """
    Detects Cross-Site Scripting vulnerabilities (CWE-79).

    Identifies:
    - innerHTML assignments with variable
    - document.write with variable
    - Unescaped template rendering
    - Response write without encoding

    Examples of vulnerable code:
        element.innerHTML = userInput
        document.write(data)
        res.send("<div>" + name + "</div>")
    """

    name = "xss"
    cwe_ids = ["CWE-79"]
    category = VulnerabilityCategory.INJECTION
    description = "Cross-Site Scripting via unescaped output"
    default_severity = Severity.MEDIUM
    supported_languages = {"javascript", "js", "typescript", "ts", "python", "php"}

    XSS_SINKS = {
        "javascript": [
            (r"\.innerHTML\s*=", "innerHTML assignment"),
            (r"\.outerHTML\s*=", "outerHTML assignment"),
            (r"document\.write\s*\(", "document.write"),
            (r"document\.writeln\s*\(", "document.writeln"),
            (r"\.insertAdjacentHTML\s*\(", "insertAdjacentHTML"),
            (r"eval\s*\(", "eval"),
        ],
        "python": [
            (r"render_template_string\s*\(", "render_template_string"),
            (r"Markup\s*\(", "Markup without escape"),
            (r"\|safe\b", "Jinja2 safe filter"),
            # Flask/Django returning HTML with f-strings
            (r'return\s+f["\'].*<[a-zA-Z]', "returning HTML with f-string"),
            (r'return\s+["\']<.*\+', "returning HTML with concatenation"),
        ],
        "php": [
            (r"echo\s+\$", "echo with variable"),
            (r"print\s+\$", "print with variable"),
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect XSS vulnerabilities."""
        if not self.enabled or not self.supports_language(language):
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        # Normalize language
        if lang_lower in ["ts", "typescript"]:
            lang_lower = "javascript"
        if lang_lower == "js":
            lang_lower = "javascript"

        sinks = self.XSS_SINKS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            for pattern, description in sinks:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if there's variable involvement
                    has_variable = bool(re.search(r"[\+\$\{]|f['\"]|\bvar\b|\blet\b|\bconst\b", line))

                    if has_variable or "innerHTML" in line:
                        findings.append(self.create_finding(
                            vulnerability_type="Cross-Site Scripting (XSS)",
                            description=f"Potential XSS via {description}",
                            line=i,
                            snippet=line.strip()[:100],
                            confidence=0.75 if has_variable else 0.6,
                            suggested_fix="Escape user input before rendering. Use textContent instead of innerHTML.",
                        ))
                        break

        return findings


@register_detector("code_injection")
class CodeInjectionDetector(BaseDetector):
    """
    Detects code injection vulnerabilities (CWE-94).

    Identifies:
    - eval() with user input
    - exec() with user input
    - Dynamic code compilation
    - Template injection

    Examples of vulnerable code:
        eval(user_input)
        exec(f"print({value})")
        compile(code, '<string>', 'exec')
    """

    name = "code_injection"
    cwe_ids = ["CWE-94"]
    category = VulnerabilityCategory.INJECTION
    description = "Code Injection via dynamic execution"
    default_severity = Severity.CRITICAL

    CODE_EXECUTION_PATTERNS = {
        "python": [
            # Any eval/exec call is dangerous if passed variable
            (r"\beval\s*\(\s*[a-zA-Z_]", "eval with variable", Severity.CRITICAL),
            (r"\bexec\s*\(\s*[a-zA-Z_]", "exec with variable", Severity.CRITICAL),
            (r"\beval\s*\([^)]*[\+\$\{]", "eval with concatenation", Severity.CRITICAL),
            (r"\bexec\s*\([^)]*[\+\$\{]", "exec with concatenation", Severity.CRITICAL),
            (r"\beval\s*\(f['\"]", "eval with f-string", Severity.CRITICAL),
            (r"\bexec\s*\(f['\"]", "exec with f-string", Severity.CRITICAL),
            (r"\bcompile\s*\([^)]+,\s*['\"]<string>['\"]", "compile", Severity.HIGH),
        ],
        "javascript": [
            # Any eval call is dangerous
            (r"\beval\s*\(\s*[a-zA-Z_]", "eval with variable", Severity.CRITICAL),
            (r"\beval\s*\([^)]*[\+\$\{]", "eval with concatenation", Severity.CRITICAL),
            (r"\bnew\s+Function\s*\([^)]*[\+\$\{]", "Function constructor", Severity.CRITICAL),
            (r"setTimeout\s*\([^)]*[\+\$\{]", "setTimeout with string", Severity.MEDIUM),
            (r"setInterval\s*\([^)]*[\+\$\{]", "setInterval with string", Severity.MEDIUM),
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect code injection vulnerabilities."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        patterns = self.CODE_EXECUTION_PATTERNS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            for pattern, description, severity in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(self.create_finding(
                        vulnerability_type="Code Injection",
                        description=f"Dynamic code execution via {description}",
                        line=i,
                        snippet=line.strip()[:100],
                        severity=severity,
                        confidence=0.9,
                        suggested_fix="Avoid eval/exec. Use safer alternatives like ast.literal_eval for Python.",
                    ))
                    break  # One finding per line

        return findings


@register_detector("xxe")
class XXEDetector(BaseDetector):
    """
    Detects XML External Entity vulnerabilities (CWE-611).

    Identifies:
    - XML parsing without disabling external entities
    - Unsafe XML parser configurations

    Examples of vulnerable code:
        etree.parse(xml_file)
        xml.etree.ElementTree.fromstring(xml)
        DocumentBuilderFactory.newInstance()  # without setFeature
    """

    name = "xxe"
    cwe_ids = ["CWE-611"]
    category = VulnerabilityCategory.INJECTION
    description = "XML External Entity (XXE) Injection"
    default_severity = Severity.HIGH
    supported_languages = {"python", "java"}

    XXE_PATTERNS = {
        "python": [
            (r"etree\.parse\s*\(", "etree.parse without defuse"),
            (r"etree\.fromstring\s*\(", "etree.fromstring without defuse"),
            (r"ET\.parse\s*\(", "ET.parse without defuse"),
            (r"ET\.fromstring\s*\(", "ET.fromstring without defuse"),
            (r"ElementTree\.parse\s*\(", "ElementTree.parse"),
            (r"ElementTree\.fromstring\s*\(", "ElementTree.fromstring"),
            (r"xml\.dom\.minidom\.parse\s*\(", "minidom.parse"),
            (r"xml\.sax\.parse\s*\(", "sax.parse"),
            (r"lxml\.etree\.parse\s*\(", "lxml.etree.parse"),
            (r"lxml\.etree\.fromstring\s*\(", "lxml.etree.fromstring"),
            (r"XMLParser\s*\(", "XMLParser without defuse"),
        ],
        "java": [
            (r"DocumentBuilderFactory\.newInstance\s*\(", "DocumentBuilderFactory without secure config"),
            (r"SAXParserFactory\.newInstance\s*\(", "SAXParserFactory without secure config"),
            (r"XMLInputFactory\.newInstance\s*\(", "XMLInputFactory without secure config"),
        ],
    }

    SAFE_PATTERNS = {
        "python": [
            r"defusedxml",
            r"resolve_entities\s*=\s*False",
        ],
        "java": [
            r"setFeature.*XMLConstants\.FEATURE_SECURE_PROCESSING",
            r"setFeature.*disallow-doctype-decl",
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect XXE vulnerabilities."""
        if not self.enabled or not self.supports_language(language):
            return []

        findings = []
        lang_lower = language.lower()

        # Check if safe patterns are present in entire code
        safe_patterns = self.SAFE_PATTERNS.get(lang_lower, [])
        has_safe_config = any(re.search(p, code, re.IGNORECASE) for p in safe_patterns)

        if has_safe_config:
            return []

        lines = code.split("\n")
        patterns = self.XXE_PATTERNS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(self.create_finding(
                        vulnerability_type="XML External Entity (XXE)",
                        description=f"Potentially unsafe XML parsing: {description}",
                        line=i,
                        snippet=line.strip()[:100],
                        confidence=0.75,
                        suggested_fix="Use defusedxml (Python) or disable external entities (Java).",
                    ))

        return findings


@register_detector("ssrf")
class SSRFDetector(BaseDetector):
    """
    Detects Server-Side Request Forgery vulnerabilities (CWE-918).

    Identifies:
    - HTTP requests with user-controlled URLs
    - URL construction with user input

    Examples of vulnerable code:
        requests.get(user_url)
        urllib.request.urlopen(url)
        fetch(userInput)
    """

    name = "ssrf"
    cwe_ids = ["CWE-918"]
    category = VulnerabilityCategory.INJECTION
    description = "Server-Side Request Forgery"
    default_severity = Severity.HIGH

    HTTP_FUNCTIONS = {
        "python": [
            "requests.get", "requests.post", "requests.put", "requests.delete",
            "requests.head", "requests.patch", "requests.request",
            "urllib.request.urlopen", "urllib2.urlopen",
            "httplib.HTTPConnection", "http.client.HTTPConnection",
            "aiohttp.ClientSession", "httpx.get", "httpx.post",
        ],
        "javascript": [
            "fetch", "axios.get", "axios.post", "axios.put",
            "http.get", "https.get", "request",
        ],
        "java": [
            "URL.openConnection", "HttpURLConnection",
            "HttpClient.execute", "RestTemplate",
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect SSRF vulnerabilities."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        functions = self.HTTP_FUNCTIONS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            for func in functions:
                func_name = func.split(".")[-1]
                pattern = rf"\b{re.escape(func_name)}\s*\("

                if re.search(pattern, line, re.IGNORECASE):
                    # Check if URL appears to be user-controlled (variable passed)
                    # Match: function(variable) or function(url) - any non-string first arg
                    has_variable = bool(re.search(
                        rf"{re.escape(func_name)}\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,\)]",
                        line
                    ))
                    # Also check for concatenation or f-strings
                    has_concat = bool(re.search(r"[\+\$\{]|f['\"]", line))

                    if has_variable or has_concat:
                        findings.append(self.create_finding(
                            vulnerability_type="Server-Side Request Forgery (SSRF)",
                            description=f"HTTP request with potentially user-controlled URL: {func}",
                            line=i,
                            snippet=line.strip()[:100],
                            confidence=0.7 if has_variable else 0.6,
                            suggested_fix="Validate and whitelist allowed URLs. Don't allow user control of full URL.",
                        ))
                        break

        return findings


@register_detector("path_traversal")
class PathTraversalDetector(BaseDetector):
    """
    Detects Path Traversal vulnerabilities (CWE-22).

    Identifies:
    - File operations with user-controlled paths
    - Lack of path sanitization

    Examples of vulnerable code:
        open(user_path)
        os.path.join(base, user_input)
        fs.readFile(userPath)
    """

    name = "path_traversal"
    cwe_ids = ["CWE-22"]
    category = VulnerabilityCategory.INPUT_VALIDATION
    description = "Path Traversal via unsanitized file paths"
    default_severity = Severity.HIGH

    FILE_OPERATIONS = {
        "python": [
            "open", "file", "os.path.join", "pathlib.Path",
            "shutil.copy", "shutil.move", "os.remove", "os.unlink",
            "os.rename", "os.listdir", "os.makedirs", "send_file",
            "send_from_directory",
        ],
        "javascript": [
            "fs.readFile", "fs.writeFile", "fs.unlink", "fs.mkdir",
            "fs.readFileSync", "fs.writeFileSync", "path.join",
            "fs.createReadStream", "fs.createWriteStream",
        ],
        "java": [
            "FileInputStream", "FileOutputStream", "FileReader",
            "FileWriter", "File", "Files.readAllBytes",
        ],
    }

    def detect(
        self,
        code: str,
        language: str,
        analysis: Optional[AnalysisResult] = None,
    ) -> List[SecurityFinding]:
        """Detect path traversal vulnerabilities."""
        if not self.enabled:
            return []

        findings = []
        lines = code.split("\n")
        lang_lower = language.lower()

        operations = self.FILE_OPERATIONS.get(lang_lower, [])

        for i, line in enumerate(lines, 1):
            for op in operations:
                op_name = op.split(".")[-1]
                pattern = rf"\b{re.escape(op_name)}\s*\("

                if re.search(pattern, line, re.IGNORECASE):
                    # Check for variable input - any non-string literal arg
                    has_variable = bool(re.search(
                        rf"{re.escape(op_name)}\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,\)]",
                        line
                    ))

                    # Also check for concatenation or f-strings
                    has_concat = bool(re.search(r"[\+\$\{]|f['\"]", line))

                    # Check for path traversal indicators
                    has_traversal = bool(re.search(r"\.\./|\.\.\\", line))

                    if has_variable or has_concat or has_traversal:
                        confidence = 0.85 if has_traversal else (0.7 if has_variable else 0.6)

                        findings.append(self.create_finding(
                            vulnerability_type="Path Traversal",
                            description=f"File operation with potentially unsafe path: {op}",
                            line=i,
                            snippet=line.strip()[:100],
                            confidence=confidence,
                            suggested_fix="Validate and sanitize file paths. Use os.path.realpath() and check against allowed directories.",
                        ))
                        break

        return findings


# Export all detectors
__all__ = [
    "SQLInjectionDetector",
    "CommandInjectionDetector",
    "XSSDetector",
    "CodeInjectionDetector",
    "XXEDetector",
    "SSRFDetector",
    "PathTraversalDetector",
]
