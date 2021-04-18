"""
    This is the setup file for the PIP minecraft_learns package.

    Written By: Kathryn Lecha and Nathan Nesbitt
    Date: 2021-01-26
"""

from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="minecraft_learns",
    version="0.1.0",
    description="Machine Learning library for Minecraft Education AI interactions",
    url="https://github.com/Nathan-Nesbitt/Minecraft_Learns",
    author=("Kathryn Lecha, Nathan Nesbitt"),
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
    ],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
