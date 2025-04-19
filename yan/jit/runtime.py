import ctypes
import os
import platform
import torch
from typing import Optional

from .template import map_ctype

# Detect Windows platform
IS_WINDOWS = platform.system() == "Windows"


class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.args = None
        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Check for required files with platform-specific library extension
        lib_ext = "kernel.dll" if IS_WINDOWS else "kernel.so"
        files = ["kernel.cu", "kernel.args", lib_ext]
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, *args) -> int:
        if self.lib is None or self.args is None:
            lib_name = os.path.join(
                self.path, "kernel.dll" if IS_WINDOWS else "kernel.so"
            )

            if IS_WINDOWS:
                # Temporarily change directory to handle DLL dependencies
                current_dir = os.getcwd()
                os.chdir(self.path)
                try:
                    self.lib = ctypes.CDLL(lib_name)
                finally:
                    os.chdir(current_dir)
            else:
                self.lib = ctypes.CDLL(lib_name)

            with open(os.path.join(self.path, "kernel.args"), "r") as f:
                self.args = eval(f.read())

        assert len(args) == len(
            self.args
        ), f"Expected {len(self.args)} arguments, got {len(args)}"

        cargs = []
        for arg, (name, dtype) in zip(args, self.args):
            if isinstance(arg, torch.Tensor):
                assert (
                    arg.dtype == dtype
                ), f"Expected tensor dtype `{dtype}` for `{name}`, got `{arg.dtype}`"
            else:
                assert isinstance(
                    arg, dtype
                ), f"Expected built-in type `{dtype}` for `{name}`, got `{type(arg)}`"
            cargs.append(map_ctype(arg))

        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))
        return return_code.value


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        if path in self.cache:
            return self.cache[path]

        if os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
