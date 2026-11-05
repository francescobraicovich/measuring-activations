import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_study import stage


class StageTest(unittest.TestCase):
    def test_reuse_and_changed_output_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.txt"
            source = root / "source.txt"
            source.write_text("input")
            command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('result')", str(output)]
            stage("example", command, [output], root, [source])
            stage("example", command, [output], root, [source])
            output.write_text("modified")
            with self.assertRaises(RuntimeError):
                stage("example", command, [output], root, [source])

    def test_failure_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                stage("failure", [sys.executable, "-c", "raise ValueError('failure')"], [root / "missing"], root)
            self.assertFalse((root / "failure.json").exists())
            self.assertIn("ValueError", (root / "failure.log").read_text())

class OrchestrationTest(unittest.TestCase):
    def test_pipeline_paths_and_gates(self):
        import json
        from unittest.mock import patch
        import pandas as pd
        from scripts import run_study

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            def simulated_stage(name, command, outputs, state_dir, inputs=()):
                stages.append(name)
                for path in outputs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.name == "metrics.csv":
                        path.write_text("model,kl\ngpt2,0.1\n")
                    elif path.name == "paired_contrasts.csv":
                        path.write_text("model,paired_kl_difference\ngpt2,0.1\n")
                    else:
                        path.write_text("synthetic fixture")
                if name.startswith("test_"):
                    gate = json.loads(Path(command[command.index("--descriptive-gate") + 1]).read_text())
                    validation = json.loads(Path(command[command.index("--validation-gate") + 1]).read_text())
                    self.assertEqual(len(gate["entries"]), 6)
                    self.assertEqual(len(validation["entries"]), 2)
                    self.assertEqual(Path(command[command.index("--calibration") + 1]),
                                     Path(next(e["calibration"] for e in gate["entries"] if
                                               e["model"] == command[command.index("--model") + 1] and
                                               e["layer"] == int(command[command.index("--layer") + 1]))))
                    self.assertTrue(all(Path(path).exists() for path in gate["figures"]))
                    self.assertTrue(Path(gate["summary"]).is_file())
                if name == "causal_figures":
                    self.assertEqual(command[-2], "--paired")
                    self.assertTrue(Path(command[-1]).is_file())
            def fake_parquet(frame, path, **kwargs):
                Path(path).write_text("synthetic fixture")
            with patch.object(run_study, "stage", side_effect=simulated_stage), \
                 patch.object(sys, "argv", ["run_study", "--output", str(root)]), \
                 patch.object(pd.DataFrame, "to_parquet", fake_parquet):
                run_study.main()
            self.assertEqual(len([name for name in stages if name.startswith("describe_")]), 6)
            self.assertEqual(len([name for name in stages if name.startswith("validation_")]), 2)
            self.assertEqual(len([name for name in stages if name.startswith("test_")]), 6)
            self.assertEqual(stages[-1], "causal_figures")
            self.assertTrue((root / "metrics.parquet").is_file())

    def test_stop_after_validation_creates_gate_without_test_stages(self):
        import json
        from unittest.mock import patch
        from scripts import run_study

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            def simulated_stage(name, command, outputs, state_dir, inputs=()):
                stages.append(name)
                for path in outputs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic fixture")
            with patch.object(run_study, "stage", side_effect=simulated_stage), \
                 patch.object(sys, "argv", ["run_study", "--output", str(root), "--stop-after-validation"]):
                run_study.main()
            gate = json.loads((root / "validation_complete.json").read_text())
            self.assertEqual({entry["model"] for entry in gate["entries"]}, {"gpt2", "smollm2"})
            self.assertEqual(len([name for name in stages if name.startswith("validation_")]), 2)
            self.assertFalse(any(name.startswith("test_") for name in stages))
            self.assertFalse((root / "test").exists())
            self.assertFalse((root / "metrics.parquet").exists())


if __name__ == "__main__":
    unittest.main()
