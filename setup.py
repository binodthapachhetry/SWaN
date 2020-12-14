import setuptools
from setuptools import setup

setup(
    name='SWaN for TIME project',
    version='0.0.2',
    description='SWaN package from private bitbucket repo',
    url='git@bitbucket.org:mhealthresearchgroup/packageswanfortime.git',
    author='Binod Thapa-Chhetry',
    author_email='binod.thapachhetry@gmail.com',
    license='unlicense',
    packages= setuptools.find_packages(),
    zip_safe=False
)