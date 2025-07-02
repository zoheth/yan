import functools
import hashlib
import os
import platform
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Tuple, Type

import cuda.bindings
import cuda.bindings.nvrtc as nvrtc
from torch.utils.cpp_extension import CUDA_HOME

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache

runtime_cache = RuntimeCache()

# Determine the OS platform
IS_WINDOWS = platform.system() == 'Windows'

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'include')

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
        return os.path.join(base_dir, 'yan')
    else:
        # Use ~/.yan on Linux/macOS
        return os.path.expanduser('~/.cache/yan')


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
        
class Compiler:
    @classmethod
    def signature(cls) -> str:
        pass

    @staticmethod
    def __version__() -> Tuple[int, int]:
        pass

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        pass

    @staticmethod
    def flags() -> List[str]:
        cpp_standard = int(os.getenv('YAN_JIT_OVERRIDE_CPP_STANDARD', 20))
        return [f'-std=c++{cpp_standard}',
                '--ptxas-options=--register-usage-level=10' +
                (',--verbose' if 'YAN_JIT_PTXAS_VERBOSE' in os.environ else ''),
                # Suppress some unnecessary warnings, such as unused variables for certain `constexpr` branch cases
                '--diag-suppress=39,161,174,177,186,940']

    @staticmethod
    def include_dirs() -> List[str]:
        return [get_jit_include_dir()]

    @classmethod
    def build(cls, name: str, code: str, runtime_cls: Type[Runtime], kwargs: Dict[str, Any] = None) -> Runtime:
        # Compiler flags
        flags = cls.flags()

        # Build signature
        enable_sass_opt = cls.__version__() <= (12, 8) and not int(os.getenv('YAN_JIT_DISABLE_FFMA_INTERLEAVE', 0))
        signature = f'{name}$${get_yan_version()}$${cls.signature()}$${flags}$${enable_sass_opt}$${code}'
        name = f'kernel.{name}.{hash_to_hex(signature)}'
        path = os.path.join(get_cache_dir(), name)

        # Check runtime cache or file system hit
        global runtime_cache
        cached_runtime = runtime_cache.get(path, runtime_cls, name, kwargs)
        if cached_runtime is not None:
            if int(os.getenv('YAN_JIT_DEBUG', 0)):
                print(f'Using cached JIT runtime {name} during build')
            return cached_runtime

        # Compile into a temporary CU file
        os.makedirs(path, exist_ok=True)
        cubin_path = os.path.join(path, 'kernel.cubin')
        tmp_cubin_path = os.path.join(make_tmp_dir(), f'nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(cubin_path)}.cubin')

        start_time = time.time()
        cls.compile(name, code, tmp_cubin_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if int(os.getenv('YAN_JIT_DEBUG', 0)):
            print(f'Compilation of JIT runtime {name} took {elapsed_time:.2f} seconds.')

        # Interleave FFMA reuse
        if enable_sass_opt:
            interleave_ffma.process(tmp_cubin_path)
            
        # Atomic replace files
        os.replace(tmp_cubin_path, cubin_path)

        # Put cache and return
        runtime = runtime_cache.get(path, runtime_cls, name, kwargs, force_enable_cache=True)
        assert runtime is not None
        return runtime


class NVCCCompiler(Compiler):
    @staticmethod
    def __version__() -> Tuple[int, int]:
        _, version = get_nvcc_compiler()
        major, minor = map(int, version.split('.'))
        return major, minor

    @classmethod
    def signature(cls) -> str:
        return f'{get_nvcc_compiler()[0]}+{cls.__version__()}'

    @classmethod
    def flags(cls) -> List[str]:
        cxx_flags = ['-fPIC', '-O3', '-fconcepts', '-Wno-deprecated-declarations', '-Wno-abi']
        return [*super().flags(), *[f'-I{d}' for d in cls.include_dirs()],
                '-gencode=arch=compute_90a,code=sm_90a',
                '-cubin', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                f'--compiler-options={",".join(cxx_flags)}']

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        # Write the code
        path = os.path.join(get_cache_dir(), name)
        src_path = os.path.join(path, 'kernel.cu')
        put(src_path, code)
        command = [get_nvcc_compiler()[0],
                   src_path, '-o', target_path,
                   *cls.flags()]
        if int(os.getenv('YAN_JIT_DEBUG', 0)) or int(os.getenv('YAN_JIT_PRINT_COMPILER_COMMAND', 0)):
            print(f'Compiling JIT runtime {name} with command {command}')

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f'NVCC compilation failed: stdout: {result.stdout}, stderr: {result.stderr}')
            assert False, f'Failed to compile {src_path}'


class NVRTCCompiler(Compiler):
    @staticmethod
    def __version__() -> Tuple[int, int]:
        res, major, minor = nvrtc.nvrtcVersion()
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            # Failed to get the actual NVRTC version, use cuda-bindings version instead
            major, minor = map(int, cuda.bindings.__version__.split('.')[:2])
        return major, minor

    @classmethod
    def signature(cls) -> str:
        return f'nvrtc+{cls.__version__()}'

    @staticmethod
    def include_dirs() -> List[str]:
        if CUDA_HOME is None:
            raise RuntimeError('CUDA_HOME is required for NVRTC compilation')
        return [get_jit_include_dir(), os.path.join(CUDA_HOME, 'include')]

    @classmethod
    def flags(cls) -> List[str]:
        flags = [*super().flags(), *[f'-I{d}' for d in cls.include_dirs()],
                 '--gpu-architecture=sm_90a', '-default-device']
        # NOTES: PCH is vital for compilation speed
        if cls.__version__() >= (12, 8):
            flags += ['--pch']
            if int(os.getenv('YAN_JIT_DEBUG', 0)):
                flags += ['--pch-verbose=true']
        return flags

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        # Create program
        code_bytes = bytes(code, 'utf-8')
        result, program = nvrtc.nvrtcCreateProgram(
            code_bytes, bytes(name, 'utf-8'), 0, [], [])
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to create program: {result}'

        # Compile
        options = [bytes(flag, 'utf-8') for flag in cls.flags()]
        if int(os.getenv('YAN_JIT_DEBUG', 0)) or int(os.getenv('YAN_JIT_PRINT_COMPILER_COMMAND', 0)):
            print(f'Compiling JIT runtime {name} with options: {options}')
        compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)[0]

        # Print compiler log
        if int(os.getenv('YAN_JIT_DEBUG', 0)) or compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to get program log size: {result}'

            log_bytes = bytes(log_size)
            result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to get program log: {result}'
            print(f'Compiler log: {log_bytes.decode("utf-8")}')

        # Exit if failed
        assert compile_result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to compile program: {compile_result}'

        # Create CUBIN
        result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to get CUBIN size: {result}'
        cubin_bytes = bytes(cubin_size)
        result = nvrtc.nvrtcGetCUBIN(program, cubin_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to get CUBIN: {result}'

        # Write into the file system
        put(target_path, cubin_bytes)

        # Destroy handler
        assert nvrtc.nvrtcDestroyProgram(program)[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, f'Failed to destroy program: {result}'


def build(name: str, code: str, runtime_cls: Type[Runtime], kwargs: Dict[str, Any] = None) -> Runtime:
    compiler_cls = NVRTCCompiler if int(os.getenv('YAN_JIT_USE_NVRTC', 0)) else NVCCCompiler
    return compiler_cls.build(name, code, runtime_cls, kwargs)