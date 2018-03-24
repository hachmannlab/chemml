from setuptools import setup, find_packages
from os import path
import cheml

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(  name = 'chemml',
        version = cheml.__version__,
        author='Mojtaba Haghighatlari, Johannes Hachmann',
        author_email='mojtabah@buffalo.edu, hachmann@buffalo.edu',
        # url='https://github.com/hachmannlab/chemml',
        project_urls={
            'Source': 'https://github.com/hachmannlab/chemml',
            'url': 'https://hachmannlab.github.io/chemml/'
        },
        description='A Machine Learning and Informatics Program Suite for the Chemical and Materials Sciences',
        long_description=long_description,

        packages = find_packages(exclude=['docs']),

        scripts = ['bin/chemmlwrapper'],

        keywords=['Machine Learning', 'Data Mining', 'Quantum Chemistry', 'Materials Science', 'Informatics'],
        license='3-Clause BSD',
        classifiers=['Development Status :: 4 - Beta',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 2.7',
                  ],

        python_requires = '>=2.7, <3',

        install_requires = ['numpy>=1.13', 'pandas>=0.20.3', 'tensorflow==1.1.0', 'keras==2.1.5',
                            'scikit-learn==0.19.1', 'babel>=2.3.4', 'matplotlib>=1.5.1', 'deap>=1.2.2',
                            'lxml','ipywidgets','graphviz'],

        test_suite='nose.collector',
        tests_require=['nose'],

        include_package_data = True,
        # package_data={
        #                 '': ['*.xyz', '*.csv', '*.vasp', '*.txt'],
        #                 # 'cheml': ['datasets/data/*', 'tests/data/*', 'tests/configfiles/*'],
        #             },
        extras_require={
                'gui': ['ipywidgets', 'graphviz'],
                'ml': ['scikit-learn>=0.18.1','tensorflow>=1.4.1', 'keras>=2.1.2'],
                'viz': ['matplotlib>=1.5.1'],
                'ga': ['deap>=1.2.2'],

            },
     )
