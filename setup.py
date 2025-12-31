"""
Setup configuration for xorl_client - Python client for XoRL training service.

This package is designed to be installed on your laptop and only includes the
client SDK for communicating with remote XoRL training servers. It does NOT
include server-side dependencies like PyTorch.
"""

from setuptools import setup, find_packages

# Read version from __init__.py
version = {}
with open("xorl_client/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Minimal dependencies - only what's needed for HTTP client
install_requires = [
    "requests>=2.28.0",
    "typing-extensions>=4.0.0",
    "numpy>=1.20.0",  # Used in types for tensor operations
]

# Optional dependencies for examples
extras_require = {
    "examples": [
        "transformers>=4.30.0",
        "wandb>=0.15.0",
    ],
    "dedicated-endpoint": [
        "openai>=1.0.0",
        
    ],
    "tokenizer": [
        "transformers>=4.45.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "black>=23.0.0",
        "mypy>=1.0.0",
    ],
}

setup(
    name="xorl-client",
    version=version.get("__version__", "0.1.0"),
    description="Python client for XoRL training service",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="XoRL Team",
    
    url="https://github.com/xorl-ai/xorl-client",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
