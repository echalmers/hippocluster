import setuptools

setuptools.setup(
    name="hippocluster",
    version="0.0.3",
    author="Eric Chalmers",
    author_email="dchalmer@ualberta.ca",
    description="Hippocluster: an efficient, brain-inspired adaptation of K-means for graph clustering",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.3",
        "networkx>=2.7.1",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
    ]
)