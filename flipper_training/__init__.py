from pathlib import Path
from lovely_tensors import monkey_patch

PACKAGE_ROOT = Path(__file__).parent
ROOT = PACKAGE_ROOT.parent

monkey_patch()
