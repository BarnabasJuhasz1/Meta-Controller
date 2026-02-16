
from setuptools import find_packages, setup

setup(
    name="oesd",
    version="1.0",
    python_requires=">=3.9",
    zip_safe=True,
    packages=find_packages(include=["oesd"]),
)