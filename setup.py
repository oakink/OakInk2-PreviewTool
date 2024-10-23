from setuptools import setup, find_packages


setup(
    name="oakink2_toolkit",
    version="0.0.4",
    python_requires=">=3.10.0",
    packages=find_packages(
        where="src",
        include="oakink2_preview*,oakink2_toolkit*",
    ),
    package_dir={"": "src"}
)
