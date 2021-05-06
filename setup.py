"""
    Installation script for alohaflow (WIP)
"""

from pathlib import Path
from sys import version_info
import re
from setuptools import setup, find_packages

requirements = ["vegasflow"]
if version_info.major >= 3 and version_info.minor >= 9:
    # For python above 3.9 the only existing TF is 2.5 which works well (even pre releases)
    tf_pack = "tensorflow"
else:
    tf_pack = "tensorflow>2.1"
requirements.append(tf_pack)
package_name = "alohaflow"
description = "Tensorized verson of ALOHA package"


def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    version_file = Path(f"src/{package_name}/__init__.py")
    initfile_lines = version_file.open("rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


setup(
    name=package_name,
    version=get_version(),
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
    python_requires=">=3.6",
    install_requires=requirements,
)
