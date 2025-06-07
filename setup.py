from setuptools import setup, find_packages

setup(
    name="deeplib",
    version="0.1.0",
    description="A deep learning framework for 3D point cloud processing",
    author="hiepbk",
    author_email="hiepbk@gmail.com",
    url="https://github.com/hiepbk/deeplib",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy",
        "pyyaml",
        "matplotlib",
        "pycocotools",
        "opencv-python",
        "tqdm",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 