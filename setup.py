from setuptools import find_packages,setup #automatically find all packages required
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if(HYPEN_E_DOT in requirements):
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(

    name = 'mlprojects',
    version='0.0.1',
    author = 'Madhav',
    author_email="madhavmaster123@gmail.com",
    packages=find_packages(),#gets all packages
    install_requires=get_requirements('requirements.txt')
)