import os
from setuptools import setup, find_packages

classifiers = [
    # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: Unix',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
]

# Get the current version number from inside the module
with open(os.path.join('ebosc', 'version.py')) as version_file:
    exec(version_file.read())

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ebosc',
    version=__version__,
    license='GPL',
    description='extended Better Oscillation Detection, implemented in python3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer = 'Julian Kosciessa',
    maintainer_email = 'kosciessa@mpib-berlin.mpg.de',
    url='https://github.com/jkosciessa/eBOSC_py/',
    packages=find_packages(),
    classifiers=classifiers,
    project_urls={
        'Issue Tracker': 'https://github.com/jkosciessa/eBOSC_py/issues',
    },
    keywords=[
        'neural rhythms', 'electrophysiology', 'EEG', 'neuroscience'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'scipy', 'pandas', 'statsmodels', 'matplotlib'
    ],
)