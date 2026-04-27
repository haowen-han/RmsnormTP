from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm_tp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="rmsnorm_tp_cpp",
            sources=[
                "csrc/bindings.cpp",
                "csrc/rmsnorm_tp.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]",
                    "--generate-code=arch=compute_100,code=[compute_100,sm_100]",
                    "-USE_NVSHMEM",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
