import setuptools
from os import path
import chemml

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='chemml',
        version=chemml.__version__,
        author='Mojtaba Haghighatlari, Johannes Hachmann',
        author_email='mojtabah@buffalo.edu, hachmann@buffalo.edu',
        # url='https://github.com/hachmannlab/chemml',
        project_urls={
            'Source': 'https://github.com/hachmannlab/chemml',
            'url': 'https://hachmannlab.github.io/chemml/'
        },
        description=
        'A Machine Learning and Informatics Program Suite for the Chemical and Materials Sciences',
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Machine Learning', 'Data Mining', 'Quantum Chemistry',
            'Materials Science', 'Drug Discovery'
        ],
        license='BSD-3C',
        packages=setuptools.find_packages(),
        include_package_data=True,

        install_requires=[
            'future', 'six',
            'numpy', 'pandas', 'scipy',
            'tensorflow', 'keras', 'h5py',
            'scikit-learn',
            'matplotlib>=1.5.1',
            'lxml', 'wget',
            'seaborn'
        ],
        extras_require={
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
                'nbsphinx'
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox'
            ],
        },
        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            # 'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        zip_safe=False,
    )

    # install_requires = ['numpy>=1.13', 'pandas>=0.20.3', 'tensorflow==1.1.0', 'keras==2.1.5',
    #                     'scikit-learn==0.19.1', 'babel>=2.3.4', 'matplotlib>=1.5.1', 'deap>=1.2.2',
    #                     'lxml','nose','ipywidgets>=7.1','widgetsnbextension>=3.1','graphviz'],

    # include_package_data = True,
    # package_data={
    #                 '': ['*.xyz', '*.csv', '*.vasp', '*.txt'],
    #                 # 'cheml': ['datasets/data/*', 'tests/data/*', 'tests/configfiles/*'],
    #             },
