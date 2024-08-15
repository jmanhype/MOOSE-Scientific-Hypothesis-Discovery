from setuptools import setup, find_packages

setup(
    name='MOOSE-Scientific-Hypothesis-Discovery',
    version='0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author='Straughter Guthrie',
    author_email='straughterguthrie@gmail.com',
    description='A project for scientific hypothesis discovery using AI.',
    url='https://github.com/jmanhype/MOOSE-Scientific-Hypothesis-Discovery',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)