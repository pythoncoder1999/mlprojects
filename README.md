first use code . to run vscode instance of folder
python -m venv .venv creates virtual environment for python
./.venv/Scripts/activate to activate environment

git init to initialize git

connect to git remote repos, add files, commit files, and push to remote repos.

create gitignore in github and select python as choice

use git pull to pull changes from github

We now create setup.py, which is responsible for creating machine learning application as a package

We also create requirements.txt, which gives all modules to use.

pip install -r requirements.txt will download all files.



components contains all the modules

pipelines runs modules sequentially





CODING



When deploying with elastic beanstalk
.ebextensions >
python.config

Must copy app.py code to application.py so application:application in config file works and remove debug option