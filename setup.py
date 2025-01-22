from setuptools import setup, find_packages

setup(
    name="transaction_balance_analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "web3>=6.0.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "tqdm>=4.65.0"  
    ],
    python_requires=">=3.7",
    author="Tao Yan, Guanda Zhao, Claudio J.Tessone",
    author_email="yan@ifi.uzh.ch, guanda.zhao@uzh.ch, claudio.tessone@uzh.ch",
    description="A tool for analyzing account balance changes in Ethereum transactions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/transaction_balance_analyzer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    package_data={
        'transaction_balance_analyzer': ['data/*.json', 'data/*.csv'],
    },
)