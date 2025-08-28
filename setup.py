from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A PyTorch implementation of ML models for image generation."

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ml-image-generators",
    version="0.1.0",
    author="David Jany",
    author_email="djany31@gmail.com",
    description="A PyTorch implementation of ML models for image generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dj-shark31/ml-image-generators.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "torch>=1.9.0",
        ],
        "full": [
            "tensorboard>=2.7.0",
            "wandb>=0.12.0",
            "pytorch-fid>=0.2.0",
            "lpips>=0.1.4",
            "scikit-image>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffusion-train=diffusion.train:main",
            "diffusion-sample=diffusion.sample:main",
            "diffusion-eval=diffusion.eval:main",
        ],
    },
    include_package_data=True,
    package_data={
        "diffusion": ["*.py"],
        "configs": ["*.py"],
    },
    keywords=[
        "diffusion",
        "ddpm",
        "generative",
        "pytorch",
        "machine-learning",
        "deep-learning",
        "image-generation",
        "ai",
        "artificial-intelligence",
    ],
    project_urls={
        "Source": "https://github.com/dj-shark31/ml-image-generators.git",
        "Documentation": "https://github.com/dj-shark31/ml-image-generators.git#readme",
    },
    zip_safe=False,
) 