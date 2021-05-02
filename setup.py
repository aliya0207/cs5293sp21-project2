from setuptools import setup, find_packages

setup(
    name='unredactor',
    version='1.0',
    author='Aliya Shaikh',
    author_email='aliyashaikh02@ou.edu',
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']	
)
