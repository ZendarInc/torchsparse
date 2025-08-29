import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

from torchsparseplusplus import __version__

version = __version__

if (torch.cuda.is_available() and CUDA_HOME is not None) or (os.getenv(
        'FORCE_CUDA', '0') == '1'):
    device = 'cuda'
    pybind_fn = f"pybind_{device}.cu"
else:
    device = 'cpu'
    pybind_fn = f"pybind_{device}.cpp"

sources = [os.path.join('torchsparseplusplus', 'backend', pybind_fn)]
for fpath in glob.glob(os.path.join('torchsparseplusplus', 'backend', '**',
                                    '*')):
    if (fpath.endswith('_cpu.cpp')
            and device in ['cpu', 'cuda']) or (fpath.endswith('_cuda.cu')
                                               and device == 'cuda'):
        sources.append(fpath)

extension_type = CUDAExtension if device == 'cuda' else CppExtension
extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': [
        '-O3', '-std=c++17', '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_89,code=sm_89',
        '-gencode=arch=compute_120,code=sm_120'
    ],
}

setup(
    name='torchsparseplusplus',
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type('torchsparseplusplus.backend',
                       sources,
                       extra_compile_args=extra_compile_args)
    ],
    install_requires=[
        'numpy',
        'backports.cached_property',
        'tqdm',
        'typing-extensions',
        'wheel',
        'rootpath',
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
