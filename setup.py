"""setup.py for backward compatibility with older pip versions.
Configuration is defined in pyproject.toml.
"""
from setuptools import setup, find_packages

setup(
    name="drone_sdk",
    version="2.0.0",
    package_dir={"": "sdk"},
    packages=find_packages("sdk"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "test": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "lint": ["flake8>=6.0.0", "mypy>=1.0.0", "black>=23.0.0"],
    },
)
