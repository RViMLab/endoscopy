import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "torch >= 1.9",
    "kornia >= 0.6"
]

setuptools.setup(
    name="endoscopy",
    version="0.0.10",
    author="Martin Huber",
    author_email="martin.huber@kcl.ac.uk",
    description="Image processing utilities for endoscopic images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RViMLab/endoscopy",
    project_urls = {
        "Bug Tracker": "https://github.com/RViMLab/endoscopy/issues"
    },
    license="MIT",
    python_revquires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=setuptools.find_packages(include=["endoscopy", "endoscopy.*"]),
    install_requires=requirements
)
