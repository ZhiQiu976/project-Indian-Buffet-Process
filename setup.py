import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IBP-Gaussian-663-2020", # Replace with your own username
    version="0.0.1",
    author="Xiaohe Yang & Zhi Qiu",
    author_email="qziuhi@126.com",
    description="A small package for using Indian Buffet Process as a prior in "
                "the Infinite linear-Gaussian binary feature model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/final-project-sta663/Indian-Buffet-Process",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
