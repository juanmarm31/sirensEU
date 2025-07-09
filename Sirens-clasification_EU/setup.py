from setuptools import setup, find_packages

PACKAGE_NAME = 'SiCaPKF'
VERSION = "0.0.1"

# Dependencies needed by this package at runtime
INSTALL_REQUIREMENTS = [
    "pyneuroner[cpu]",
    'tensorflow>2.12.0',
    'spacy>2.0.18',
    'audiomentations',
    'numpy',
    'matplotlib',
    'seaborn',
    'scikit-learn'
]

# Dependencies needed for this setup.py script to run
SETUP_REQUIREMENTS = []

# Additional dependencies needed to run the tests
TEST_REQUIREMENTS = []

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    url='https://gitlab.tu-ilmenau.de/juro7695/Sirens-clasification_EU.git',  
    license='MIT',  # Specify the correct license
    author='Juan Manuel Rodriguez',
    author_email='juan.ma.rm31@gmail.com',

    description='A package for siren classification using CNN and CRNN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    install_requires=INSTALL_REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,

    packages=find_packages(),

    package_data={
        '': ['README.md'],
    },

    scripts=['scripts/SiCaPKF'],  # Ensure this path is correct

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11.8',
    ],
)
