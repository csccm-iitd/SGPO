from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='LOGOS_GPO',
    version='0.1',
    description='Gaussian Process operator with PyTorch and GPyTorch',
    author='sk',
    # author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),  # Reads the dependencies from requirements.txt
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
