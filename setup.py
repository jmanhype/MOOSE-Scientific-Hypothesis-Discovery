from setuptools import setup, find_packages

setup(
    name='MOOSE-Scientific-Hypothesis-Discovery',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'dspy',
        'python-dotenv',
        'aiohttp',
        'cachetools',
        'backoff',
        'nest_asyncio',
        'sentence-transformers',
        'rouge',
        'openai',
        'pydub',
    ],
    entry_points={
        'console_scripts': [
            'moose=main:main',
        ],
    },
)
