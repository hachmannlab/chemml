from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(  name='chemml',
        version='0.4.1',
        author='Mojtaba Haghighatlari',
        author_email='mojtabah@buffalo.edu',
        url='https://bitbucket.org/hachmanngroup/chemml',
        project_urls={
            'Source': 'https://bitbucket.org/hachmanngroup/chemml',
            'github': 'https://github.com/mojtabah/ChemML',
            'url': 'https://mojtabah.github.io/ChemML/'
        },
        description='A Machine Learning and Informatics Program Suite for the Chemical and Materials Sciences',
        long_description=long_description,

        packages = find_packages(exclude=['docs', 'rst_generator']),

        scripts = ['bin/cheml', 'chemmlwrapper'
                   ],

        keywords=['Machine Learning', 'Data Mining', 'Quantum Chemistry', 'Materials Science', 'Informatics'],
        license='3-Clause BSD',
        classifiers=['Development Status :: 4 - Beta',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 2.7',
                   'License :: OSI Approved :: 3-Clause BSD License',
                  ],

        install_requires = ['numpy>=1.13', 'pandas>=0.20.3'],
        extras_require={
                'gui': ['ipywidgets', 'graphviz'],
            },
     )
