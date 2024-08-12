from setuptools import setup, find_packages

setup(name="gptquery",
      version="0.0.1",
      author="Alex Havrilla",
      author_email="alexdahoas@gmail.com",
      packages=find_packages(),
      install_requires=["litellm"])