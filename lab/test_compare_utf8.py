"""Replay preparation preserves UTF-8 text, including on legacy Windows locales."""
import builtins
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_compare():
    # Input preparation does not need the optional HTTP client or tokenizer.
    spec = importlib.util.spec_from_file_location("compare_utf8", Path(__file__).with_name("compare.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"aiohttp": types.ModuleType("aiohttp"),
                                 "tokenizers": types.SimpleNamespace(Tokenizer=object)}):
        spec.loader.exec_module(module)
    return module


class Prepared(Exception):
    """Stop after preparing inputs, before starting either benchmark arm."""


class ReplayEncodingTests(unittest.TestCase):
    def check_replay(self, legacy_encoding=None):
        compare = load_compare()
        corpus = [{"messages": [{"role": "user", "content": "été — 한글 中文 🚀"},
                                {"role": "assistant", "content": "réponse\n二行目"}], "cuts": [1, 2]},
                  {"messages": [{"role": "user", "content": "café"},
                                {"role": "assistant", "content": "привет"}], "cuts": [1, 2]}]
        recipe = {"card": "fixture", "model": "fixture", "weights": "fixture@revision", "set": {"pool": 128}}
        launch = {"ctx": 128, "seqs": 1, "args": ["--max-total-tokens", "128"]}
        tok = types.SimpleNamespace(encode=lambda text: types.SimpleNamespace(ids=list(text)))
        real_open = builtins.open
        streams = []

        def open_replay(file, *args, **kwargs):
            if Path(file) != replay:
                return real_open(file, *args, **kwargs)
            # Reproduce a non-UTF-8 default on every CI platform. Explicit encodings win.
            if legacy_encoding is not None:
                kwargs.setdefault("encoding", legacy_encoding)
            stream = real_open(file, *args, **kwargs)
            streams.append(stream)
            return stream

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            replay = root / "replay.jsonl"
            replay.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in corpus) + "\n", encoding="utf-8")
            argv = ["compare.py", str(recipe_path), str(recipe_path), "--replay", str(replay), "--levels", "1"]
            try:
                with patch.object(sys, "argv", argv), \
                     patch.object(compare.lab, "render", return_value=launch), \
                     patch.object(compare, "tokenizer", return_value=tok), \
                     patch.object(compare, "workload", wraps=compare.workload) as workload, \
                     patch.object(compare, "boot", side_effect=Prepared) as boot, \
                     patch.object(compare, "open", side_effect=open_replay, create=True):
                    with self.assertRaises(Prepared):
                        compare.main()
                self.assertEqual(workload.call_args.args[0], corpus)
                boot.assert_called_once()
                self.assertTrue(all(stream.closed for stream in streams))
            finally:
                for stream in streams:
                    stream.close()

    def test_replay_uses_utf8_with_cp1252_default(self):
        self.check_replay("cp1252")

    def test_replay_uses_utf8_with_cp949_default(self):
        self.check_replay("cp949")

    def test_replay_uses_utf8_with_native_default(self):
        self.check_replay()


if __name__ == "__main__":
    unittest.main()
