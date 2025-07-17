import os
import setuptools
import shutil
import subprocess
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

current_dir = os.path.dirname(os.path.realpath(__file__))
third_party_include_dirs = (
    "third-party/cutlass/include/cute",
    "third-party/cutlass/include/cutlass",
    # "third-party/flashinfer/include/flashinfer",
)


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()
        self.make_jit_include_symlinks_specific()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(current_dir, "yan/include", dirname)
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)

    @staticmethod
    def make_jit_include_symlinks_specific():
        tk_dir = "third-party/ThunderKittens"
        src_dir = os.path.join(current_dir, tk_dir, "include")
        dst_dir = os.path.join(current_dir, "yan/include", "tk")
        assert os.path.exists(src_dir)
        if os.path.exists(dst_dir):
            assert os.path.islink(dst_dir)
            os.unlink(dst_dir)
        os.symlink(src_dir, dst_dir, target_is_directory=True)



class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Then run the regular build
        build_py.run(self)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, "yan/include")
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


setuptools.setup(
    cmdclass={
        "develop": PostDevelopCommand,
        "build_py": CustomBuildPy,
    },
)

