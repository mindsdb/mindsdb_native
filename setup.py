import setuptools
import sys
import os


def remove_requirements(requirements, name, replace=None):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements

sys_platform = sys.platform

about = {}
with open("mindsdb_native/__about__.py") as fp:
    exec(fp.read(), about)

long_description = open('README.md', encoding='utf-8').read()

with open('requirements.txt', 'r') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

with open('requirements_test.txt', 'r') as req_file:
    test_requirements = [req.strip() for req in req_file.read().splitlines()]

snowflake_requirements = []
with open('optional_requirements_snowflake.txt', 'r') as fp:
    for line in fp:
        snowflake_requirements.append(line.rstrip('\n'))

extra_data_sources_requirements = []
with open('optional_requirements_extra_data_sources.txt', 'r') as fp:
    for line in fp:
        extra_data_sources_requirements.append(line.rstrip('\n'))

dependency_links = []

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    url=about['__github__'],
    download_url=about['__pypi__'],
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require = {
        'extra_data_sources': extra_data_sources_requirements
        ,'snowflake': snowflake_requirements
    },
    tests_require = test_requirements,
    dependency_links=dependency_links,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.6"
)
