import os
import importlib
from setuptools import setup, find_packages

def readme():
    readme_path = os.path.join(os.path.dirname(os.path.normpath(__file__)), "README.md")
    with open(readme_path, "r") as f:
        content = f.read()
    return content

def get_dep():
    req_txt = os.path.join(os.path.dirname(os.path.normpath(__file__)), "req_toolkit.txt")
    with open(req_txt, "r") as f:
        content = f.read()

    res_default = [el for el in content.split("\n") if len(el) > 0 and "@git+" not in el]
    res_thirdparty = [el for el in content.split("\n") if len(el) > 0 and "@git+" in el]

    res_final = []
    for el in res_thirdparty:
        pkg = el.split("@git+")[0]
        try:
            importlib.import_module(pkg)
        except ImportError:
            res_final.append(el)

    res_final.extend(res_default)
    return res_final

setup(
    name="oakink2_toolkit",
    version="0.0.5",
    author="Xinyu Zhan",
    author_email="kelvin34501@foxmail.com",
    description="OakInk2 Toolkit",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    packages=find_packages(
        where="src",
        include="oakink2_toolkit*,oakink2_preview*",
    ),
    package_dir={"": "src"},
    install_requires=get_dep(),
    # script binary
    entry_points={
        'console_scripts': [
            'oakink2_viz_gui = oakink2_preview.launch.viz.gui:main',
            'oakink2_viz_seg3d = oakink2_preview.launch.viz.seg_3d:main',
        ],
    },
)
