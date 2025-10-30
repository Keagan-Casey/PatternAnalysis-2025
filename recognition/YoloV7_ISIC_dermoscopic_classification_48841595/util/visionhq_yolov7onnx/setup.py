from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.readlines()

setup(
    name="visionhq_yolov7onnx",
    version="2.1.4",
    description="Wrapper for a YOLOv7ONNX model.",
    author="Allen Liang, Daniel Zhang, Duval Longa",
    author_email="hello@visionhq.io",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
