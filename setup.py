from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dsh",
    version="0.1.0",
    authorname ="vincentiusyoshua",
    description="A hashing implementation with multiple security levels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vincentiusyoshua/dynamic_security_hash",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "blake3>=0.3.0",
        "cryptography>=3.4.0",
        "psutil>=5.8.0",
    ],
)
