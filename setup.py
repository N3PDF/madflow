"""
    Installation script for madflow (WIP)
"""

from pathlib import Path
from sys import version_info
import re
import os
from setuptools import setup, find_packages

requirements = ["vegasflow", "pdfflow"]
if version_info.major >= 3 and version_info.minor >= 9:
    # For python above 3.9 the only existing TF is 2.5 which works well (even pre releases)
    tf_pack = "tensorflow"
else:
    tf_pack = "tensorflow>2.1"
requirements.append(tf_pack)
package_name = "madflow"
package_root = "python_package"
repository_root = Path(__file__).parent

description = "Package for GPU fixed order calculations"
long_description = (repository_root / "readme.md").read_text()


def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    version_file = repository_root / f"{package_root}/{package_name}/__init__.py"
    initfile_lines = version_file.open("rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    return "unknown"


setup(
    name=package_name,
    version=get_version(),
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GNU GPLv3",
    author="S. Carrazza, J. Cruz-Martinez, M. Rossi, M. Zaro",
    author_email="https://github.com/N3PDF/madflow/issues/new",
    url="https://github.com/N3PDF/madflow/",
    package_dir={"": package_root},
    packages=find_packages(package_root),
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
    entry_points={
        "console_scripts": [
            "madflow = madflow.scripts.madflow_exec:main",
        ]
    },
)
