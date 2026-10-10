from setuptools import find_packages, setup

setup(
    name="mconvert",
    version="0.1.0",
    description="GUI structure-format converter built on mkits' structure engine",
    packages=find_packages(include=["mconvert", "mconvert.*"]),
    package_data={"mconvert": ["resources/*"]},
    include_package_data=True,
    install_requires=[
        "PySide6>=6.5",
        "numpy>=1.24",
        "spglib>=2.0",
    ],
    entry_points={
        "console_scripts": ["mconvert = mconvert.app:main"],
    },
    python_requires=">=3.9",
)
