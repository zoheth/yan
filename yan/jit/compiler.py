import hashlib
import functools
import os
import re
import subprocess
import uuid
from torch.utils.cpp_extension import CUDA_HOME
from typing import Tuple

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return f'{os.path.dirname(os.path.abspath(__file__))}/../include'

@functools.lru_cache(maxsize=None)
def get_yan_version() -> str:
    base_include_dir = get_jit_include_dir()
    assert os.path.exists(base_include_dir), f'Cannot find Yan include directory {base_include_dir}'
    
    md5 = hashlib.md5()
    
    cuh_files = []
    for root, _, files in os.walk(base_include_dir):
        for file in files:
            if file.endswith('.cuh'):
                cuh_files.append(os.path.join(root, file))
    
    for file_path in sorted(cuh_files):
        with open(file_path, 'rb') as f:
            md5.update(f.read())

    with open(f'{os.path.dirname(os.path.realpath(__file__))}/interleave_ffma.py', 'rb') as f:
        md5.update(f.read())
    
    return md5.hexdigest()[0:12]

@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv('YAN_NVCC_COMPILER'):
        paths.append(os.getenv('YAN_NVCC_COMPILER'))
    paths.append(f'{CUDA_HOME}/bin/nvcc')

    # Try to find the first available NVCC compiler
    least_version_required = '12.3'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f'{path} --version').read())
            version = match.group(1)
            assert match, f'Cannot get the version of NVCC compiler {path}'
            assert version >= least_version_required, f'NVCC {path} version {version} is lower than {least_version_required}'
            return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if 'YAN_CACHE_DIR' in os.environ:
        path = os.getenv('YAN_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser('~') + '/.yan'


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return f'{get_default_user_dir()}/tmp'


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return f'{get_default_user_dir()}/cache'


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data, is_binary=False):
    # Write and do POSIX atomic replace
    tmp_file_path = f'{make_tmp_dir()}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}'
    with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
        f.write(data)
    os.replace(tmp_file_path, path)


def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    # Compiler flags
    nvcc_flags = ['-std=c++17', '-shared', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                  '-gencode=arch=compute_89,code=sm_89',
                  '--ptxas-options=--register-usage-level=10' + (',--verbose' if 'YAN_PTXAS_VERBOSE' in os.environ else ''),
                  # Suppress some unnecessary warnings, such as unused variables for certain `constexpr` branch cases
                  '--diag-suppress=177,174,940']
    cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations', '-Wno-abi']
    flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]

    # Build signature
    enable_sass_opt = get_nvcc_compiler()[1] <= '12.8' and int(os.getenv('YAN_DISABLE_FFMA_INTERLEAVE', 0)) == 0
    signature = f'{name}$${get_yan_version()}$${code}$${get_nvcc_compiler()}$${flags}$${enable_sass_opt}'
    name = f'kernel.{name}.{hash_to_hex(signature)}'
    path = f'{get_cache_dir()}/{name}'

    # Check runtime cache or file system hit
    global runtime_cache
    if runtime_cache[path] is not None:
        if os.getenv('YAN_JIT_DEBUG', None):
            print(f'Using cached JIT runtime {name} during build')
        return runtime_cache[path]
    
    # Write the code
    os.makedirs(path, exist_ok=True)
    args_path = f'{path}/kernel.args'
    src_path = f'{path}/kernel.cu'
    put(args_path, ', '.join([f"('{arg_def[0]}', {typename_map[arg_def[1]]})" for arg_def in arg_defs]))
    put(src_path, code)

    # Compile into a temporary SO file
    so_path = f'{path}/kernel.so'
    tmp_so_path = f'{make_tmp_dir()}/nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so'

    # Compile
    command = [get_nvcc_compiler()[0],
               src_path, '-o', tmp_so_path,
               *flags,
               *[f'-I{d}' for d in include_dirs]]
    if os.getenv('YAN_JIT_DEBUG', None) or os.getenv('YANJIT_PRINT_NVCC_COMMAND', False):
        print(f'Compiling JIT runtime {name} with command {command}')
    return_code = subprocess.check_call(command)
    assert return_code == 0, f'Failed to compile {src_path}'

    # Atomic replace SO file
    os.replace(tmp_so_path, so_path)

    # Put cache and return
    runtime_cache[path] = Runtime(path)
    return runtime_cache[path]
