# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

## Python Environment

Install latest version of `miniconda` from [here](https://docs.conda.io/en/latest/miniconda.html).

The python environment configuration is stored in `environment.yaml`. To recreate it in `conda`:

```
conda env create -f environment.yaml
``` 

This will create an environment called `bevo` with all required packages. `CUDA` support might need to be configured separately depending on the machine you're running it on.
