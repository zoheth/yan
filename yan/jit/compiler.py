import hashlib
import functools
import os
import re
import subprocess
import uuid
import platform
import shutil
from torch.utils.cpp_extension import CUDA_HOME
from typing import Tuple

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()

# Determine the OS platform
IS_WINDOWS = platform.system() == 'Windows'

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/../include')

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

    interleave_path = os.path.normpath(f'{os.path.dirname(os.path.realpath(__file__))}/interleave_ffma.py')
    with open(interleave_path, 'rb') as f:
        md5.update(f.read())
    
    return md5.hexdigest()[0:12]

@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv('YAN_NVCC_COMPILER'):
        paths.append(os.getenv('YAN_NVCC_COMPILER'))
    
    # Windows-specific path
    if IS_WINDOWS:
        if CUDA_HOME:
            paths.append(os.path.join(CUDA_HOME, 'bin', 'nvcc.exe'))
        # Check common install paths on Windows
        program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
        for cuda_version in ['v12.3', 'v12.4', 'v12.5', 'v12.6', 'v12.7', 'v12.8']:
            paths.append(os.path.join(program_files, 'NVIDIA GPU Computing Toolkit', 'CUDA', cuda_version, 'bin', 'nvcc.exe'))
    else:
        # Linux path
        if CUDA_HOME:
            paths.append(os.path.join(CUDA_HOME, 'bin', 'nvcc'))

    # Try to find the first available NVCC compiler
    least_version_required = '12.3'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            try:
                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                match = version_pattern.search(output)
                if match:
                    version = match.group(1)
                    if version >= least_version_required:
                        return path, version
                    else:
                        print(f'NVCC {path} version {version} is lower than {least_version_required}')
                else:
                    print(f'Cannot determine version for NVCC compiler {path}')
            except (subprocess.SubprocessError, OSError) as e:
                print(f'Error checking NVCC version for {path}: {e}')
                continue
    
    raise RuntimeError('Cannot find any available NVCC compiler with version >= 12.3. '
                      'Please make sure CUDA Toolkit 12.3 or higher is installed and set CUDA_HOME '
                      'or YAN_NVCC_COMPILER environment variable.')


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if 'YAN_CACHE_DIR' in os.environ:
        path = os.getenv('YAN_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    
    if IS_WINDOWS:
        # Use %LOCALAPPDATA% on Windows, which is typically C:\Users\<username>\AppData\Local
        base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return os.path.join(base_dir, '.yan')
    else:
        # Use ~/.yan on Linux/macOS
        return os.path.expanduser('~/.yan')


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return os.path.join(get_default_user_dir(), 'tmp')


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return os.path.join(get_default_user_dir(), 'cache')


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data, is_binary=False):
    # Create a temporary file
    tmp_file_path = os.path.join(make_tmp_dir(), 
                               f'file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}')
    
    # Write data to temporary file
    with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
        f.write(data)
    
    # Use atomic replacement if possible, otherwise fallback to regular replace
    try:
        if IS_WINDOWS:
            # On Windows, os.replace works but requires destination to not exist
            # If it exists, we need to remove it first to avoid PermissionError
            if os.path.exists(path):
                os.unlink(path)
        os.replace(tmp_file_path, path)
    except OSError:
        # Fallback method if atomic replace fails
        shutil.copy2(tmp_file_path, path)
        os.unlink(tmp_file_path)


def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    # Get the extension based on platform
    lib_ext = '.dll' if IS_WINDOWS else '.so'
    
    # Base compiler flags
    common_flags = ['-std=c++17', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                   '-gencode=arch=compute_89,code=sm_89',
                   '--use_fast_math']
    
    # Platform-specific flags
    if IS_WINDOWS:
        nvcc_flags = [*common_flags, 
                     '-shared', 
                     '--ptxas-options=--register-usage-level=10',
                     '--diag-suppress=177,174,940']
        # Windows MSVC options differ from gcc/clang
        cxx_flags = ['/MD', '/O2', '/GR', '/EHsc', '/wd4819']
        flags = [*nvcc_flags, f'-Xcompiler={",".join(cxx_flags)}']
    else:
        # Linux flags
        nvcc_flags = [*common_flags, 
                     '-shared',
                     '--ptxas-options=--register-usage-level=10' + 
                     (',--verbose' if 'YAN_PTXAS_VERBOSE' in os.environ else ''),
                     '--diag-suppress=177,174,940']
        cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations', '-Wno-abi']
        flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    
    include_dirs = [get_jit_include_dir()]

    # Build signature
    enable_sass_opt = get_nvcc_compiler()[1] <= '12.8' and int(os.getenv('YAN_DISABLE_FFMA_INTERLEAVE', 0)) == 0
    signature = f'{name}$${get_yan_version()}$${code}$${get_nvcc_compiler()}$${flags}$${enable_sass_opt}$${IS_WINDOWS}'
    name = f'kernel.{name}.{hash_to_hex(signature)}'
    path = os.path.join(get_cache_dir(), name)

    # Check runtime cache or file system hit
    global runtime_cache
    if runtime_cache[path] is not None:
        if os.getenv('YAN_JIT_DEBUG', None):
            print(f'Using cached JIT runtime {name} during build')
        return runtime_cache[path]
    
    # Write the code
    os.makedirs(path, exist_ok=True)
    args_path = os.path.join(path, 'kernel.args')
    src_path = os.path.join(path, 'kernel.cu')
    put(args_path, ', '.join([f"('{arg_def[0]}', {typename_map[arg_def[1]]})" for arg_def in arg_defs]))
    put(src_path, code)

    # Compile into a temporary lib file
    lib_path = os.path.join(path, f'kernel{lib_ext}')
    tmp_lib_path = os.path.join(make_tmp_dir(), 
                              f'nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(lib_path)}{lib_ext}')

    # Compile
    command = [get_nvcc_compiler()[0],
               src_path, '-o', tmp_lib_path,
               *flags,
               *[f'-I{d}' for d in include_dirs]]
    
    if os.getenv('YAN_JIT_DEBUG', None) or os.getenv('YANJIT_PRINT_NVCC_COMMAND', False):
        print(f'Compiling JIT runtime {name} with command {command}')
    
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Failed to compile {src_path}: {e}')

    # Atomic replace lib file if possible
    try:
        if IS_WINDOWS and os.path.exists(lib_path):
            # On Windows, need to remove the existing file first
            os.unlink(lib_path)
        os.replace(tmp_lib_path, lib_path)
    except OSError:
        # Fallback if atomic replace fails
        shutil.copy2(tmp_lib_path, lib_path)
        os.unlink(tmp_lib_path)

    # Put cache and return
    runtime_cache[path] = Runtime(path)
    return runtime_cache[path]