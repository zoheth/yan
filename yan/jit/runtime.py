import os
import subprocess
import time
import torch
import cuda.bindings.driver as cbd

from typing import Any, Dict, Optional, Type
from torch.utils.cpp_extension import CUDA_HOME

class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        files = ['kernel.cubin']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, **kwargs) -> int:
        if self.kernel is None:
            start_time = time.time_ns()

            path = bytes(os.path.join(self.path, 'kernel.cubin'), 'utf-8')
            result, self.lib = cbd.cuLibraryLoadFromFile(path, [], [], 0, [], [], 0)
            assert result == cbd.CUresult.CUDA_SUCCESS, f'Failed to load library: {result}'

            command = [f'{CUDA_HOME}/bin/cuobjdump', '-symbols', path]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert result.returncode == 0
            illegal_names = ['vprintf', '__instantiate_kernel', '__internal', '__assertfail']
            check_illegal = lambda line: any([name in line for name in illegal_names])
            kernel_names = [line.split()[-1] for line in result.stdout.splitlines()
                            if line.startswith('STT_FUNC') and not check_illegal(line)]
            assert len(kernel_names) == 1, f'Too many kernels in the library: {kernel_names}'

            result, self.kernel = cbd.cuLibraryGetKernel(self.lib, bytes(kernel_names[0], encoding='utf-8'))
            assert result == cbd.CUresult.CUDA_SUCCESS, f'Failed to load kernel: {result}'

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1e6
            if int(os.getenv('YAN_JIT_DEBUG', 0)):
                print(f'Loading JIT runtime {self.path} took {elapsed_time:.2f} ms.')

        return self.launch(self.kernel, kwargs)
    
    def __del__(self) -> None:
        if self.lib is not None:
            res = cbd.cuLibraryUnload(self.lib)[0]
            if res != cbd.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to unload library {self.path}: {res}')

class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __setitem__(self, path: str, runtime: Runtime) -> None:
        self.cache[path] = runtime

    def get(self, path: str, runtime_cls: Type[Runtime],
            name: str ='', kwargs: Dict[str, Any] = None,
            force_enable_cache: bool = False) -> Optional[Runtime]:
        if path in self.cache:
            return self.cache[path]
        
        use_cache = force_enable_cache or not int(os.getenv('YAN_JIT_DISABLE_CACHE', 0))
        if use_cache and os.path.exists(path) and Runtime.is_path_valid(path):
            if name and (int(os.getenv('YAN_JIT_DEBUG', 0)) or int(os.getenv('YAN_PRINT_CONFIGS', 0))):
                simplified_kwargs = dict()
                for key, value in kwargs.items() if kwargs is not None else dict().items():
                    value = f'torch.Tensor<{value.dtype}>' if isinstance(value, torch.Tensor) else value
                    value = f'cuda.bindings.driver.CUtensorMap' if isinstance(value, cbd.CUtensorMap) else value
                    simplified_kwargs[key] = value
                print(f'Put kernel {name} with {simplified_kwargs} into runtime cache')
                
            runtime = runtime_cls(path)
            self.cache[path] = runtime
            return runtime
        return None
            