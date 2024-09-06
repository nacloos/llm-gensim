from setuptools import setup, find_packages


setup(
    name='llm_gensim',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('llm_gensim')],
    install_requires=[
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "python-dotenv",
        "anthropic",
        "google-generativeai",
        "openai",
    ],
    description='',
    author=''
)
