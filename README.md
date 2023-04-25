# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

## Python Environment

Install latest version of `miniconda` from [here](https://docs.conda.io/en/latest/miniconda.html).

The python environment configuration is stored in `environment.yaml`. To recreate it in `conda`:

```
conda env create -f environment.yaml
``` 

This will create an environment called `bevo` with all required packages. `CUDA` support might need to be configured separately depending on the machine you're running it on.

`environment.yml` has specific build information. If you are running into dependency or package version issues, use `environment_nobuild.yml`, and conda will resolve all dependencies to your specific machine.

To push to the repo and commit changes, run `gh auth login`, and supply a GitHub [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
