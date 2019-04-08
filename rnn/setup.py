from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy>=1.16']

setup(
    name='trainer',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
