import os
from setuptools import setup, find_packages

classifiers = [
    # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
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

setup(
    name='ebosc',
    version='0.95-dev',
    license='LGPL-3.0-or-later',
    description='extended Better Oscillation Detection, implemented in python3',
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