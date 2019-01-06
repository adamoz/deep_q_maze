import sys
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(
    name='qrl_navigation',
    version='0.1',
    description='Deep reinforcement learing solver of unity banana maze',
    author='adamoz',
    author_email='',
    url='',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    scripts=[],
    setup_requires=['pytest-runner'],
    install_requires=[
        'numpy==1.15.2',
        'tqdm==4.19.5',
        'click==6.7',
        'matplotlib==2.1.1',
        'pandas==0.22.0',
        'pillow==5.0.0',
        'pytest==3.3.1',
        'scikit-learn==0.19.1',
        'scipy==1.0.0',
        'seaborn==0.9.0',
        'sklearn==0.0',
        'tqdm==4.19.5',
        'torch==0.4.0',
        'torchvision==0.2.1',
        'ipython==6.2.1',
    ],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    include_package_data=True,
    package_data={
        # Add all files under data/ directory.
        # This data will be part of this package.
        # Access them with pkg_resources module.
        # Folder with data HAVE TO be in some module, so dont add it to folder with tests, which SHOULD NOT be a module.
    },
)
