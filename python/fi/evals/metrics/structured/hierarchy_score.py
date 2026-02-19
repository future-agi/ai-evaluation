"""
Hierarchy Score Metric.

Tree-based structural comparison using edit distance concepts.
Inspired by STED (Structural Tree Edit Distance) for comparing hierarchical structures.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base_metric import BaseMetric
from .types import JSONInput, StructuredInput
from .validators import JSONValidator


class HierarchyScore(BaseMetric[JSONInput]):
    """
    Evaluates structural similarity using tree-based comparison.

    Compares the structural hierarchy of response vs expected:
    - Tree structure matching
    - Key path similarity
    - Depth alignment
    - Array structure matching

    Inspired by STED (Structural Tree Edit Distance) concepts for
    comparing hierarchical JSON structures.

    Score: 0.0 (completely different structure) to 1.0 (identical structure)

    Example:
        >>> metric = HierarchyScore()
        >>> result = metric.evaluate([{
        ...     "response": '{"user": {"name": "Alice"}, "items": [1, 2]}',
        ...     "expected": {"user": {"name": "Bob", "email": "b@ex.com"}, "items": [1, 2, 3]}
        ... }])
        # Compares structure, not values
    """

    @property
    def metric_name(self) -> str:
        return "hierarchy_score"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()
        self.key_weight = self.config.get("key_weight", 0.5)
        self.type_weight = self.config.get("type_weight", 0.3)
        self.depth_weight = self.config.get("depth_weight", 0.2)
        self.max_depth = self.config.get("max_depth", 10)

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response
        expected = inputs.expected

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if expected is None:
            return {"output": 0.0, "reason": "No expected structure provided"}

        # Parse response
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {"output": 0.0, "reason": "Cannot parse response"}

        parsed = syntax_result.parsed

        # Build structure fingerprints
        expected_structure = self._build_structure(expected, "", 0)
        actual_structure = self._build_structure(parsed, "", 0)

        # Compare structures
        similarity = self._compare_structures(expected_structure, actual_structure)

        return {
            "output": round(similarity["overall"], 4),
            "reason": f"Structure similarity: keys={similarity['key_similarity']:.2f}, types={similarity['type_similarity']:.2f}, depth={similarity['depth_similarity']:.2f}",
            "key_similarity": round(similarity["key_similarity"], 4),
            "type_similarity": round(similarity["type_similarity"], 4),
            "depth_similarity": round(similarity["depth_similarity"], 4),
            "expected_depth": expected_structure["max_depth"],
            "actual_depth": actual_structure["max_depth"],
            "missing_keys": list(similarity["missing_keys"]),
            "extra_keys": list(similarity["extra_keys"]),
            "parsed": parsed,
        }

    def _build_structure(
        self,
        data: Any,
        path: str,
        depth: int,
    ) -> Dict[str, Any]:
        """Build structure fingerprint for comparison."""
        structure = {
            "keys": set(),
            "types": {},
            "depths": {},
            "max_depth": depth,
            "array_shapes": {},
        }

        if depth > self.max_depth:
            return structure

        if isinstance(data, dict):
            for key, value in data.items():
                key_path = f"{path}.{key}" if path else key
                structure["keys"].add(key_path)
                structure["types"][key_path] = self._get_type_name(value)
                structure["depths"][key_path] = depth

                # Recurse into nested structures
                nested = self._build_structure(value, key_path, depth + 1)
                structure["keys"].update(nested["keys"])
                structure["types"].update(nested["types"])
                structure["depths"].update(nested["depths"])
                structure["max_depth"] = max(structure["max_depth"], nested["max_depth"])
                structure["array_shapes"].update(nested["array_shapes"])

        elif isinstance(data, list):
            structure["array_shapes"][path] = len(data)
            if data:
                # Sample first element for structure
                sample_path = f"{path}[]"
                structure["keys"].add(sample_path)
                structure["types"][sample_path] = self._get_type_name(data[0])
                structure["depths"][sample_path] = depth

                if isinstance(data[0], (dict, list)):
                    nested = self._build_structure(data[0], sample_path, depth + 1)
                    structure["keys"].update(nested["keys"])
                    structure["types"].update(nested["types"])
                    structure["depths"].update(nested["depths"])
                    structure["max_depth"] = max(structure["max_depth"], nested["max_depth"])
                    structure["array_shapes"].update(nested["array_shapes"])

        return structure

    def _get_type_name(self, value: Any) -> str:
        """Get JSON-like type name."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _compare_structures(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two structure fingerprints."""
        expected_keys = expected["keys"]
        actual_keys = actual["keys"]

        # Key similarity (Jaccard)
        if expected_keys or actual_keys:
            intersection = expected_keys & actual_keys
            union = expected_keys | actual_keys
            key_similarity = len(intersection) / len(union) if union else 1.0
        else:
            key_similarity = 1.0

        # Type similarity for matching keys
        matching_keys = expected_keys & actual_keys
        if matching_keys:
            type_matches = sum(
                1 for k in matching_keys
                if expected["types"].get(k) == actual["types"].get(k)
            )
            type_similarity = type_matches / len(matching_keys)
        else:
            type_similarity = 0.0 if expected_keys else 1.0

        # Depth similarity
        max_expected = expected["max_depth"]
        max_actual = actual["max_depth"]
        if max_expected == 0 and max_actual == 0:
            depth_similarity = 1.0
        else:
            depth_similarity = 1.0 - abs(max_expected - max_actual) / max(max_expected, max_actual, 1)

        # Overall weighted score
        overall = (
            self.key_weight * key_similarity +
            self.type_weight * type_similarity +
            self.depth_weight * depth_similarity
        )

        return {
            "overall": overall,
            "key_similarity": key_similarity,
            "type_similarity": type_similarity,
            "depth_similarity": depth_similarity,
            "missing_keys": expected_keys - actual_keys,
            "extra_keys": actual_keys - expected_keys,
        }


class TreeEditDistance(BaseMetric[JSONInput]):
    """
    Computes normalized tree edit distance between structures.

    Based on simplified tree edit distance where operations are:
    - Insert node
    - Delete node
    - Rename node (change key or value)

    Score: 0.0 (identical) to 1.0 (completely different)
    Note: This returns DISTANCE, so lower is better.
    """

    @property
    def metric_name(self) -> str:
        return "tree_edit_distance"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = JSONValidator()
        self.insert_cost = self.config.get("insert_cost", 1.0)
        self.delete_cost = self.config.get("delete_cost", 1.0)
        self.rename_cost = self.config.get("rename_cost", 0.5)

    def compute_one(self, inputs: JSONInput) -> Dict[str, Any]:
        response = inputs.response
        expected = inputs.expected

        if not response or not response.strip():
            return {"output": 1.0, "reason": "Empty response (max distance)"}

        if expected is None:
            return {"output": 1.0, "reason": "No expected structure provided"}

        # Parse response
        syntax_result = self.validator.validate_syntax(response)
        if not syntax_result.syntax_valid:
            return {"output": 1.0, "reason": "Cannot parse response"}

        parsed = syntax_result.parsed

        # Compute tree operations needed
        operations = self._compute_operations(expected, parsed, "$")

        # Calculate total cost
        total_cost = sum(op["cost"] for op in operations)

        # Normalize by tree size
        expected_size = self._count_nodes(expected)
        actual_size = self._count_nodes(parsed)
        max_size = max(expected_size, actual_size, 1)

        normalized_distance = min(total_cost / max_size, 1.0)

        return {
            "output": round(normalized_distance, 4),
            "reason": f"{len(operations)} operations (cost={total_cost:.2f})",
            "operations": operations[:10],  # Limit for readability
            "total_operations": len(operations),
            "total_cost": round(total_cost, 4),
            "expected_nodes": expected_size,
            "actual_nodes": actual_size,
            "parsed": parsed,
        }

    def _compute_operations(
        self,
        expected: Any,
        actual: Any,
        path: str,
    ) -> List[Dict[str, Any]]:
        """Compute edit operations needed to transform actual to expected."""
        operations = []

        # Handle type mismatches
        if type(expected) != type(actual):
            operations.append({
                "type": "replace",
                "path": path,
                "from_type": type(actual).__name__,
                "to_type": type(expected).__name__,
                "cost": self.rename_cost,
            })
            return operations

        if isinstance(expected, dict):
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())

            # Missing keys (need insert)
            for key in expected_keys - actual_keys:
                operations.append({
                    "type": "insert",
                    "path": f"{path}.{key}",
                    "cost": self.insert_cost,
                })

            # Extra keys (need delete)
            for key in actual_keys - expected_keys:
                operations.append({
                    "type": "delete",
                    "path": f"{path}.{key}",
                    "cost": self.delete_cost,
                })

            # Recurse into common keys
            for key in expected_keys & actual_keys:
                operations.extend(
                    self._compute_operations(expected[key], actual[key], f"{path}.{key}")
                )

        elif isinstance(expected, list):
            # Simple length-based comparison for arrays
            len_diff = abs(len(expected) - len(actual))
            for i in range(len_diff):
                if len(expected) > len(actual):
                    operations.append({
                        "type": "insert",
                        "path": f"{path}[{len(actual) + i}]",
                        "cost": self.insert_cost,
                    })
                else:
                    operations.append({
                        "type": "delete",
                        "path": f"{path}[{len(expected) + i}]",
                        "cost": self.delete_cost,
                    })

            # Compare common elements
            for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                operations.extend(
                    self._compute_operations(exp_item, act_item, f"{path}[{i}]")
                )

        else:
            # Scalar comparison
            if expected != actual:
                operations.append({
                    "type": "rename",
                    "path": path,
                    "from": str(actual)[:50],
                    "to": str(expected)[:50],
                    "cost": self.rename_cost,
                })

        return operations

    def _count_nodes(self, data: Any) -> int:
        """Count total nodes in tree."""
        if isinstance(data, dict):
            return 1 + sum(self._count_nodes(v) for v in data.values())
        elif isinstance(data, list):
            return 1 + sum(self._count_nodes(item) for item in data)
        else:
            return 1
