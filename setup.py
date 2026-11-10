from setuptools import setup, find_packages

setup(
    name="logos-gpo",
    version="1.0.0",
    description="LoGoS-GPO: scalable Gaussian process operators with neural-operator means (PyTorch)",
    author="Sawan Kumar, Tapas Tripura, Rajdip Nayek, Souvik Chakraborty",
    url="https://github.com/csccm-iitd/SGPO",
    license="MIT",
    packages=find_packages(include=["sgpo", "sgpo.*"]),
    install_requires=["torch>=2.1", "numpy", "scipy", "h5py", "PyWavelets", "pytorch_wavelets", "ptwt", "matplotlib"],
    extras_require={
        "data": ["scikit-fem", "triangle", "pandas", "nbformat", "nbclient", "ipykernel", "gdown"],
        "test": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
