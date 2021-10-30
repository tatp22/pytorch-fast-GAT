import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast_gat",
    version="0.1.0",
    author="Peter Tatkowski",
    author_email="tatp22@gmail.com",
    description="A PyTorch implementation of Graph Attention Networks, with experimental speedup features.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tatp22/pytorch-fast-GAT",
    packages=setuptools.find_packages(),
    keywords=['attention', 'deep learning', 'artificial intelligence', 'sparse attention'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        'torch'
    ],
)
