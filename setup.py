from setuptools import setup, find_packages

classifiers = [
    # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)'
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
]

setup(
    name='eBOSC-py',
    version='0.0.1',
    license='LGPL-3.0-or-later',
    description='extended Better Oscillation Detection, implemented in python3',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    maintainer = 'Julian Kosciessa',
    maintainer_email = 'kosciessa@mpib-berlin.mpg.de',
    url='https://github.com/jkosciessa/eBOSC_py/',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=classifiers,
    project_urls={
        'Issue Tracker': 'https://github.com/jkosciessa/eBOSC_py/issues',
    },
    keywords=[
        'neural rhythms', 'electrophysiology', 'EEG', 'neuroscience'
    ],
    python_requires='>=3.6',
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
)
