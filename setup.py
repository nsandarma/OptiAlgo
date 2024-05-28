from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as req_file:
    requirements = req_file.read().splitlines()

DESCRIPTION = "OptiAlgo menyediakan solusi cepat dan andal untuk mencari algoritma terbaik bagi pengguna, serta memberikan fleksibilitas dalam menangani berbagai masalah data."

NAME = "optialgo"
VERSION = "1.0.0"
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nsandarma/OptiAlgo",
    author="nsandarma",
    author_email="nsandarma@gmail.com",
    license="MIT",
    packages=find_packages(
        exclude=["testing", "dataset_ex", "env", "images", "examples"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
