from setuptools import setup, find_packages

setup(
    name="codelib",
    version="0.1",
    author="Johan Stax Jakobsen",
    author_email="jsj.fi@cbs.dk",
    description=("Helper functions for the course"
                 "'Python for the Financial Economist'"),
    packages=find_packages(),
    classifiers=['Programming Language :: Python :: 3'],
    python_requires='=>3.8',
    install_requires=['matplotlib==3.3.3', 'numpy==1.19.5', 'pandas==1.2.1', 'pandas-datareader==0.10.0',
                      'Quandl==3.6.1', 'scipy==1.6.1', 'seaborn==0.11.1']
)