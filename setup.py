from pathlib import Path

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tfvit",
    version="2.1.0",
    description="Keras v3 (all backends) reimplementation of ViT models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shkarupa-alex/tfvit",
    author="Shkarupa Alex",
    author_email="shkarupa.alex@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
