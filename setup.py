from setuptools import setup, find_packages

setup(
    name="traffic-violation-system",
    version="1.0.0",
    description="Automated traffic violation detection, classification, and evidence pipeline.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
