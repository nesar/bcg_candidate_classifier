"""
Setup script for BCG Deployment Package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="bcg-deploy",
    version="0.1.0",
    author="BCG Classifier Team",
    description="Deployment package for BCG (Brightest Cluster Galaxy) classification models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "scipy>=1.7.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'bcg-inference=scripts.run_inference:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="astronomy bcg galaxy-clusters machine-learning",
)
