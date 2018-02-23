from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        '': ['*.yaml']
    },
    description='Recommendation trainer application'
)
