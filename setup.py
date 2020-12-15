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
    zip_safe=False,
    install_requires=['scikit-learn==0.22.2.post1','pandas==1.1.3','scipy==1.5.2','numpy==1.19.2']
)