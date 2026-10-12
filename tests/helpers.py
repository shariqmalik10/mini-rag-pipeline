import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_KB = ROOT / "kb"
SAMPLE_QUERIES = ROOT / "queries.json"
FIXTURES = ROOT / "tests" / "fixtures"


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
