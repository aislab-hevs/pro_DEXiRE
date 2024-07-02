# DEXiRE_pro

DEXiRE_pro extends DEXiRE rule-based explanations to include unstructured data, uncertainty and probability reasoning.

<!-- trunk-ignore(markdownlint/MD033) -->
<p align="center">
<img src="images/logo/logo.png" alt="logo_dexire" width="100"/>
</p>

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and license info here --->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project aims to produce probabilistic explanations of heterogeneous data with probabilistic graphical models.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a `<Windows/Linux/Mac>` machine.
- You have installed version of `python 3.9` or create an environment with python 3.9 version.
- It is recommended to create an environment with conda or venv to isolate the execution and avoid libraries version conflict.

## Installing DEXiRE_pro

To install DEXiRE_pro, follow these steps:

Linux and macOS:

```
python -m pip install --upgrade setuptools
```

Windows:

```
python -m pip install --upgrade setuptools
```

Then in the root directory of this repository execute the following command with the environment activated:

```
pip install .
```

Or using the following command in the root directory of this repository:

```
python setup.py install
```

## Installing with wheels

The package can be compile to a wheel file which is easy to install with python package manager pip. To build a wheel execute the following command in the root directory of this repository:

For Unix/Linux/macOS build:

```
python3 -m pip install --upgrade build
python3 -m build
```

For Windows:

```
py -m pip install --upgrade build
py -m build
```

The wheel installer can be found in the dist/ subdirectory. Position yourself in the dist/ subdirectory execute the following command:

```
pip install dexire_pro_0_0_1.whl
```

The wheel installer (.whl file) can be distributed to install in other environments.

## Using DEXiRE_pro

TODO: Add content here. And explain how to use DEXiRE_pro


## Contributing to DEXiRE_pro

To contribute to DEXiRE_pro, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request and describe the contribution.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Project manager 

This project have been developed under the supervision of:

- Davide Calvaresi

## Contributors

Thanks to the following people who have contributed to this project:

- [@victorc365](https://github.com/victorc365) 📖

## Acknowledge  

<!--TODO: Add paper citation--> 
This work is supported by the Chist-Era grant CHIST-ERA19-XAI-005, and by the Swiss National Science Foundation (G.A. 20CH21_195530). 

## Contact

If you want to contact me you can reach me at <victorc365@gmail.com>.

## License

<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT](https://opensource.org/license/mit).