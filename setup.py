import setuptools
from setuptools import setup

setup(
    name='SWaN for TIME project',
    version='0.8',
    description='SWaN package from private bitbucket repo',
    url='git@bitbucket.org:mhealthresearchgroup/packageswanfortime.git',
    author='Binod Thapa-Chhetry',
    author_email='binod.thapachhetry@gmail.com',
    license='unlicense',
    packages= setuptools.find_packages(),
    zip_safe=False,
    install_requires=['scikit-learn==0.23.2'],
    package_data={'SWaN':['StandardScalar_all_data.sav','LogisticRegression_all_data_F1score_0.70.sav']},
    include_package_data=True
)