# Conditional Musical Accompaniment Generation
AI 535 Deep Learning Final Project

## Enviroment
To avoid version control issues, use the following commands to load the python enviroment for this project:
```bash
module load python/3.10 cuda/11.7
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
If you add pakages please run before committing:
```bash
pip freeze > requirements.txt
```

## Files
### slakh-utils
For easier version control, the `slakh-utils` is a submodule from this [repo](https://github.com/shawn120/slakh-utils/tree/4118ea16222d11d295496845e898cd497c7b7673). To update this submodule:

```bash
git submodule init
git submodule update
```
read more about submodules:

[A submodule tutorial from Bitbucket](https://www.atlassian.com/git/tutorials/git-submodule)

[Submodule intro from official website](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

## Data Split
### Subset convention:
Slakh2100-orig: original split.

Slakh2100-split2: move files to make sure no duplicate songs in train and test set. Can be transferd by `splits_v2.json`

Slakh2100-redux: Remove the duplicate songs. Can be transferd by `redux.json`