#!/usr/bin/env python
# encoding: utf-8

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os.path
from setuptools import find_packages, setup

SETUP_PATH = os.path.dirname(os.path.abspath(__file__))


setup(
    name="wcontrib",
    author="William Chris Beard",
    author_email="wcbeard10@gmail.com",
    description="A Python library for Mozilla Data Science code snippets",
    url="https://github.com/wcbeard/wcontrib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "scipy"],
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "git_describe_command": os.path.join(
            SETUP_PATH, "describe_revision.py"
        )
    },
    extras_require={},
)
