from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="paligemma-pytorch",
    version="0.1.0",
    author="",
    author_email="",
    description="A PyTorch implementation of PaliGemma: A 3B Vision-Language Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "wandb": [
            "wandb>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "paligemma-train=src.training.train:main",
            "paligemma-inference=src.inference.predict:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "pytorch",
        "deep-learning",
        "vision-language-model",
        "paligemma",
        "multimodal",
        "computer-vision",
        "natural-language-processing",
        "machine-learning",
        "artificial-intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your usearname/repo/issues",
        "Source": "https://github.com/username/repo.git",
        "Documentation": "https://github.com/username/repo#readme",
    },
)