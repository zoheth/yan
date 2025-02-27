import argparse
import mmap
import os
import re
import subprocess
from torch.utils.cpp_extension import CUDA_HOME

def run_cuobjdump(file_path):
    command = [f'{CUDA_HOME}/bin/cuobjdump', '-sass', file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    return result.stdout

