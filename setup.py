"""
    Installation script for alohaflow (WIP)
"""

from setuptools import setup, find_packages

requirements = ['vegasflow']
if version_info.major >=3 and version_info.minor >= 9:
    # For python above 3.9 the only existing TF is 2.5 which works well (even pre releases)
    tf_pack = "tensorflow"
else:
    tf_pack = "tensorflow>2.1"
requirements.append(tf_pack)
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
    python_requires='>=3.6',
    install_requires=requirements,
)
