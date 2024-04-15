from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Requirements for installing the package
REQUIREMENTS = [
'tensorflow == 2.15.0',
'pandas == 2.2.1',
'scikit-learn == 1.4.1.post1',
'sympy == 1.12',
'pytest == 8.1.1',
'matplotlib == 3.8.3',
'seaborn == 0.13.2',
'yellowbrick',
'graphviz == 0.20.3',
'dexire @ https://github.com/aislab-hevs/DEXiRE/releases/download/0.0.1/dexire-0.0.1-py3-none-any.whl',
'numpy == 1.26.4',
'd3blocks == 1.4.9',
'bnlearn == 0.8.7'
]

# Some details 
CLASSIFIERS = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
]

setup(
    name='pro_dexire',
    version='0.0.1',
    description='Deep Explanation and Rule Extractor (DEXiRE)\
        is a rule extractor explainer to explain Deep learning modules\
        through rule sets. DEXiTR has been extended to used PRObabilistic graphical models.',
    author='Victor Hugo Contreras and Davide Calvaresi',
    author_email='victorc365@gmail.com',
    packages=find_packages(),
    install_dependencies=REQUIREMENTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.9',
)