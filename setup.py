"""
    Installation script for alohaflow (WIP)
"""

from setuptools import setup, find_packages


requirements = ["tensorflow", "vegasflow"]
package_name = "alohaflow"
description = "Tensorized verson of ALOHA package"

version = "0.01"  # TODO

setup(
    name=package_name,
    version=version,
    description=description,
    author="S. Carrazza, J. Cruz-Martinez, M. Rossi, M. Zaro",
    author_email="https://github.com/N3PDF/pyoutformg5amc/issues/new",
    url="https://github.com/N3PDF/pyoutformg5amc/",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=requirements,
)
