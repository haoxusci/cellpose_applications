"""Set up file for NEW_REPO package."""
from pathlib import Path

from setuptools import find_packages, setup

PROJECT_DIR = Path(__file__).parent.resolve()
README_FILE = PROJECT_DIR / "README.md"
LONG_DESCR = README_FILE.read_text(encoding="utf-8")
VERSION = (
    (PROJECT_DIR / "cellpose_applications" / "VERSION").read_text().strip()
)
GITHUB_URL = "https://github.com/haoxusci/cellpose_applications"
DOWNLOAD_URL = f"{GITHUB_URL}/archive/master.zip"

requirements = []
try:
    with open("requirements.txt", "r") as fd:
        requirements = [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    print("WARNING: missing requirements.txt.")

setup(
    name="cellpose_applications",
    version=VERSION,
    description="a repo for applications of cellpose",
    long_description=LONG_DESCR,
    long_description_content_type="text/markdown",
    author="Hao Xu",
    author_email="xuhaonju@gmail.com",
    url=GITHUB_URL,
    download_url=DOWNLOAD_URL,
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
    license="Apache-2.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
