# Pro_dexire

Pro_dexire extends DEXiRE explanations to include unstructured data, uncertainty and probability reasoning.

<!-- trunk-ignore(markdownlint/MD033) -->
<p align="center">
<img src="images/logo/logo.png" alt="logo_dexire" width="100"/>
</p>

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and license info here --->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project aims to provide recipes querying chatGPT models.

## Prerequisites

Before you begin, ensure you have met the following requirements:

<!--- These are just example requirements. Add, duplicate or remove as required --->

- You have a `<Windows/Linux/Mac>` machine.
- You have installed version of `python 3.9` or create an environment with python 3.9 version.
- It is recommended to create an environment with conda or venv to isolate the execution and avoid libraries version conflict.
- You have an API_KEY for querying the GPT models through API calls.

## Installing RecipeGPT

To install RecipeGPT, follow these steps:

Linux and macOS:

```
python -m pip install --upgrade setuptools
```

Windows:

```
python -m pip install --upgrade setuptools
```

In the root directory RecipeGPT execute the following command with the active environment  activated:

```
pip install .
```

Or in the main folder with the environment activated  execute the following command in the terminal:

```
python setup.py install
```

## Installing with wheels

The package can be compile to a wheel fire and the easy installed. To build a wheel execute the following command in the terminal and localized in the RecipeGPT main folder:

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

The wheel installer will be appear in the dist subdirectory. Localize in the dist subdirectory execute the following command:

```
pip install dexire-0.0.1-py3-none-any.whl
```

The wheel installer (.whl file) cna be distributed to install in other environments.

## Using RecipeGPT

TODO: Add content here. 


## Contributing to RecipeGPT

<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->

To contribute to RecipeGPT, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

- [@victorc365](https://github.com/victorc365) 📖

## Acknowledge  

TODO: set a papaer to cite maybe one on chatGPT

## Contact

If you want to contact me you can reach me at <victorc365@gmail.com>.

## License

<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT](https://opensource.org/license/mit).