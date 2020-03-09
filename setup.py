import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gymCICY",
    version="0.1",
    author="Robin Schneider",
    author_email="robin.schneider@physics.uu.se",
    description="A gym environment for exploring heterotic line bundle models on CICYs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-schneider/gymCICY",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "chainer",
        "chainerrl",
        "pyCICY>=0.3",
        "gym",
    ],
)