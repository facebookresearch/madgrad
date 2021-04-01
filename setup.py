# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="madgrad",
    version="1.1",
    author="Aaron Defazio",
    author_email="adefazio@fb.com",
    description="A general purpose PyTorch Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/madgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
