from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the extension module
extension = CUDAExtension(
    name='vegas_cuda_extension.cuda_extension',
    sources=[
        'src/vegas_cuda_extension/cuda_wrapper.cu',
    ],
    extra_compile_args={
        'cxx': ['-g'],
        'nvcc': ['-O3', '-m64'],
    },
    extra_link_args=['-lcuda']
)

# Setup the package
setup(
    name='vegas_cuda_extension',
    package_dir={"": "src"},
    package_data={'': ['cuda_kernel.ptx']},
    packages=find_packages(where='src'),
    include_package_data=True,
    ext_modules = [extension],
    include_dirs=['cuda-samples/Common/'],
    cmdclass={'build_ext': BuildExtension}
)
