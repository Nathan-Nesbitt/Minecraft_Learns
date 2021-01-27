"""
    This is the setup file for the PIP minecraft_learns package.

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from setuptools import setup

setup(
    name="minecraft_learns",
    version="0.0.1",
    description=
        "Machine Learning library for Minecraft Education AI interactions",
    url="https://github.com/Nathan-Nesbitt/Minecraft_Learns",
    author=(
        "Carlos Rueda Carrasco, Kathryn Lecha, Nathan Nesbitt, " + 
        "Adrian Morillo Quiroga"),
    packages=[
        "minecraft_learns"
    ],
    install_requires=[
        "pandas",
        "numpy",
    ],
    zip_safe=False
)
