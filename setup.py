"""Setup script for fashion generation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-generation",
    version="1.0.0",
    author="Fashion Generation Team",
    author_email="team@fashion-generation.com",
    description="Modern GAN-based fashion item generation using Fashion-MNIST",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fashion-generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pre-commit>=3.3.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-train=scripts.train:main",
            "fashion-sample=scripts.sample:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fashion_generation": ["configs/*.yaml", "configs/**/*.yaml"],
    },
)
