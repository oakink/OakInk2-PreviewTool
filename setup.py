from setuptools import setup, find_packages


setup(
    name="oakink2_preview",
    version="0.0.3",
    python_requires=">=3.10.0",
    packages=find_packages(
        where="src",
        include="oakink2_preview*",
    ),
    package_dir={"": "src"}
)
