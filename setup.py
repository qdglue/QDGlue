#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

# TODO(btjanaka): Figure out what deps need to go in the readthedocs config.
install_requires = [
    "numpy>=1.20.0",  # >=1.20.0 is when numpy.typing becomes available.
    "gymnasium>=0.26.0",
]

extras_require = {
    "torch": [
        # TODO(ryanboldi): Specify versions here.
        "torch",
    ],
    "jax": [
        # TODO(looka): Specify versions here.
        "jax",
        "jaxlib",
    ],
    # All dependencies except for dev. Don't worry if there are duplicate
    # dependencies, since setuptools automatically handles duplicates.
    "all": [
        # TODO: Specify all dependencies here.
    ],
    "dev": [
        "pip>=20.3",
        "pylint",
        "black>=23.1.0",
        "isort",
        "pre-commit",
        # Testing
        "pytest==7.0.1",
        "pytest-cov==3.0.0",
        "pytest-benchmark==3.4.1",
        "pytest-xdist==2.5.0",
        # Documentation
        "mkdocs==1.4.3",
        "mkdocs-material==9.1.19",
        "mkdocstrings[python]==0.22.0",
        "mkdocs-gen-files==0.5.0",
        "mkdocs-literate-nav==0.6.0",
        # Distribution
        "bump2version==1.0.1",
        "wheel==0.40.0",
        "twine==4.0.2",
        "check-wheel-contents==0.4.0",
    ],
}

setup(
    author="QDGlue team",
    author_email="rieffelj@union.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="QD Glue Benchmarking Suite",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="qdglue",
    name="qdglue",
    packages=find_packages(include=["qdglue", "qdglue.*"]),
    python_requires=">=3.7.0",
    test_suite="tests",
    url="https://github.com/qdglue/qdglue",
    version="0.0.0",
    zip_safe=False,
)
