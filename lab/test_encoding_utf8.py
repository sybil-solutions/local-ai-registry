import ast
import pathlib
import unittest


class ExplicitUtf8EncodingTest(unittest.TestCase):
    def test_all_lab_read_write_text_specify_encoding(self):
        lab_dir = pathlib.Path(__file__).resolve().parent
        missing = []
        for path in sorted(lab_dir.rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("read_text", "write_text")
                ):
                    if not any(k.arg == "encoding" for k in node.keywords):
                        rel = path.relative_to(lab_dir.parent)
                        missing.append(f"{rel}:{node.lineno}:{node.func.attr}")
        self.assertEqual(missing, [], f"Missing explicit encoding= in: {missing}")


if __name__ == "__main__":
    unittest.main()
