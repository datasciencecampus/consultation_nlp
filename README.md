<img src="https://github.com/datasciencecampus/awesome-campus/blob/master/ons_dsc_logo.png">

# `consultation_nlp`
[![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#experimental)
[![codecov](https://codecov.io/gh/datasciencecampus/consultation_nlp/branch/main/graph/badge.svg?token=bvdkp2cCG8)](https://codecov.io/gh/datasciencecampus/consultation_nlp)
[![Twitter](https://img.shields.io/twitter/url?label=Follow%20%40DataSciCampus&style=social&url=https%3A%2F%2Ftwitter.com%2FDataSciCampus)](https://twitter.com/DataSciCampus)

Python code for preliminary natural language processing analysis for 2023 population transformation
consultation

```{warning}
Where this documentation refers to the root folder we mean where this README.md is
located.
```

## Getting started

To start using this project, [first make sure your system meets its
requirements](requirements.txt).

It's suggested that you install this pack and it's requirements within a virtual environment.

### Cloning the repo
To clone the repo, open command promt and navigate to the directory you want to save the repo to and call:
`git clone https://github.com/datasciencecampus/consultation_nlp.git`

### Pre-commit actions
This repository contains a configuration of pre-commit hooks. These are language agnostic and focussed on repository security (such as detection of passwords and API keys). If approaching this project as a developer, you are encouraged to install and enable `pre-commits` by running the following in your shell:
   1. Install `pre-commit`:

      ```
      pip install pre-commit
      ```
   2. Enable `pre-commit`:

      ```
      pre-commit install
      ```

### Installing the package (Python Only)

Whilst in the root folder, in the command prompt, you can install the package and it's dependencies
using:

```shell
python -m pip install -U pip setuptools
pip install -e .
```
or use the `make` command:
```shell
make install
```

This installs an editable version of the package. Meaning, when you update the
package code, you do not have to reinstall it for the changes to take effect.
(This saves a lot of time when you test your code)

Remember to update the setup and requirement files inline with any changes to your
package. The inital files contain the bare minimum to get you started.

### Running the pipeline (Python only)

The entry point for the pipeline is stored within the package and called `run_pipeline.py`.
To run the pipeline, run the following code in the terminal (whilst in the root directory of the
project).

```shell
python src/run_pipeline.py
```

Alternatively, most Python IDE's allow you to run the code directly from the IDE using a `run` button.


### Running the streamlit app (dashboard)

1) Ensure all requirements are downloaded from the requirements.txt by openning up the shell terminal (anaconda prompt) and running:
```shell
pip install -r requirements.txt
```
2) Keep the shell terminal open and navigate to the directory where this code is saved and run:
```shell
streamlit run streamlit_app.py
```
## Licence

This codebase is released under the MIT License. This covers both the codebase and any sample code in
the documentation. The documentation is ©Crown copyright and available under the terms of the
Open Government 3.0 licence.

## Contributing

[If you want to help us build, and improve `consultation_nlp`, view our
contributing guidelines](docs/contributor_guide/CONTRIBUTING.md).

### Requirements

[```Contributors have some additional requirements!```](docs/contributor_guide/CONTRIBUTING.md)

- Python 3.6.1+ installed
- a `.secrets` file with the [required secrets and
  credentials](#required-secrets-and-credentials)
- [load environment variables][docs-loading-environment-variables] from `.env`

To install the contributing requirements, open your terminal and enter:
```shell
python -m pip install -U pip setuptools
pip install -e .[dev]
pre-commit install
```
or use the `make` command:
```shell
make install_dev
```

## Acknowledgements

[This project structure is based on the `govcookiecutter` template
project][govcookiecutter].

[contributing]: https://github.com/best-practice-and-impact/govcookiecutter/blob/main/%7B%7B%20cookiecutter.repo_name%20%7D%7D/docs/contributor_guide/CONTRIBUTING.md
[govcookiecutter]: https://github.com/best-practice-and-impact/govcookiecutter
[docs-loading-environment-variables]: https://github.com/best-practice-and-impact/govcookiecutter/blob/main/%7B%7B%20cookiecutter.repo_name%20%7D%7D/docs/user_guide/loading_environment_variables.md
[docs-loading-environment-variables-secrets]: https://github.com/best-practice-and-impact/govcookiecutter/blob/main/%7B%7B%20cookiecutter.repo_name%20%7D%7D/docs/user_guide/loading_environment_variables.md#storing-secrets-and-credentials
