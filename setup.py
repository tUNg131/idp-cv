from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Dummy detection for IDP'
LONG_DESCRIPTION = 'A package that allows to detect dummies for IDP projects'

# Setting up
setup(
    name="idp206",
    version=VERSION,
    author="Tung Xuan Le",
    author_email="<tl526@cam.ac.uk>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-contrib-python', 'numpy', 'sshtunnel'],
    keywords=['python', 'opencv'],
    classifiers=[
        "Development Status :: 6 - Deployment",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ]
)