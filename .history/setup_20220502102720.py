# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements_test.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="servier",
    version="0.2",
    # packages=["servier"],
    description="ML Test LBC",
    author="Vincent Haguet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[required_packages],
    entry_points={
        "console_scripts": [
            "train = main:train",
            "predict = servier.main:predict",
            "evaluate = servier.main:evaluate",
        ],
    },
)
