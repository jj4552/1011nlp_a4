"""Static wiring tests run in a fresh extraction without ML dependencies."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

class Static(unittest.TestCase):
    def test_all_python_parse(self):
        for p in ROOT.rglob('*.py'):
            with self.subTest(file=str(p.relative_to(ROOT))):ast.parse(p.read_text(encoding='utf8'))
    def test_all_critical_scripts_exist(self):
        for name in ('run_a4.sbatch','run_pipeline.py','preflight.py','setup_data.py',
                     'part1/main.py','part1/utils.py','part2/t5_engine.py',
                     'part2/sql_eval.py','part2/q4_q6.py','part2/evaluate.py','smoke_models.py'):
            self.assertTrue((ROOT/name).exists(),name)
    def test_cli_help_works_offline(self):
        for name in ('part1/main.py','part2/t5_engine.py','part2/q4_q6.py',
                     'setup_data.py','preflight.py','smoke_models.py'):
            with self.subTest(script=name):
                cp=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True)
                self.assertEqual(cp.returncode,0,cp.stderr)
                self.assertIn('usage:',cp.stdout)
    def test_entrypoint_wiring(self):
        run=(ROOT/'run_pipeline.py').read_text()
        for stage in ('q1_model','q2_eval','q3_train','q4','t5_train','t5_dev','t5_test','q6'):
            self.assertIn(stage,run)
        s=(ROOT/'run_a4.sbatch').read_text()
        self.assertIn('run_pipeline.py',s)
        self.assertIn('--gres=gpu:1',s)

if __name__=='__main__':unittest.main(verbosity=2)