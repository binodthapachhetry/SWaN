import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SWaN_accel",
    version="1.9",
    author="binodtc",
    author_email="binod.thapachhetry@gmail.com",
    description="A pacakge to classify sleep-wear, wake-wear, and non-wear in accelerometer dataset.",
    url="https://bitbucket.org/mhealthresearchgroup/packageswanfortime.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=['scikit-learn==0.23.2'],
    package_data={'SWaN_accel':['StandardScalar_all_data.sav','LogisticRegression_all_data_F1score_0.70.sav']},

    include_package_data=True
)