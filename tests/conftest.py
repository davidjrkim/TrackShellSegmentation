"""
Stub out heavyweight dependencies (torch, torchvision) that are not installed
in the local venv (they live in the Docker image only).  This allows test
collection and patching to succeed without a 2 GB install.
"""
import sys
from unittest.mock import MagicMock

for _mod in (
    "torch",
    "torch.nn",
    "torch.cuda",
    "torch.no_grad",
    "torchvision",
    "torchvision.models",
    "torchvision.models.segmentation",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
