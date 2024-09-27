from setuptools import setup

install_requires = [
    "numpy",
    "pandas",
    "pytest",
    "scipy",
    "frozendict",
]


setup(
    name="wlstar",
    install_requires=install_requires,
    version="1.0",
    scripts=[],
    packages=["wlstar"],
)