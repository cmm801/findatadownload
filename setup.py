from setuptools import setup, find_packages

setup(
    name='findatadownload',
    version='0.1.0',
    author='Christopher Miller',
    author_email='cmm801@gmail.com',
    packages=find_packages(),
    #package_dir={'findatadownload' : 'findatadownload'},
    include_package_data=True,
    scripts=[],
    url='http://pypi.python.org/pypi/findatadownload/',
    license='MIT',
    description='A package for downloading financial data.',
    long_description=open('README.md').read(),
    install_requires=[
        'coinbasepro',
        'numpy',
        'pandas',
        'pandas_datareader',
        'requests-cache',
        'setuptools-git',
        'yfinance',
    ],
)
