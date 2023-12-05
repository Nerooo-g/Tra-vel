from packaging import version
import torch

assert version.parse(torch.__version__) >= version.parse("2.0.0")
