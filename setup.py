from pathlib import Path

import setuptools

VERSION = "0.0.1"

NAME = "mids"

INSTALL_REQUIRES = [
    "numpy>=2.2.1",
    "scipy>=1.15.0",
    "networkx[default]>=3.4.2"
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="Compute the Approximate Independent Dominating Set for undirected graph encoded in DIMACS format.",
    url="https://github.com/frankvegadelgado/mids",
    project_urls={
        "Source Code": "https://github.com/frankvegadelgado/mids",
        "Documentation Research": "https://github.com/frankvegadelgado/mids",
    },
    author="Frank Vega",
    author_email="vega.frank@gmail.com",
    license="MIT License",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires=">=3.12",
    # Requirements
    install_requires=INSTALL_REQUIRES,
    packages=["mids"],
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'mid = mids.app:main',
            'test_mid = mids.test:main',
            'batch_mid = mids.batch:main'
        ]
    }
)