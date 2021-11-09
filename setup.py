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
            'tensorflow',
            'h5py',
            'scikit-learn',
            'matplotlib>=1.5.1',
            'lxml', 'wget',
            'seaborn',
            'openpyxl', 'ipywidgets'
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
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
    )
