# coding=utf-8
from setuptools import setup

setup(
    name="postpanda_helper",
    version="0.0.2",
    packages=["postpanda_helper"],
    install_requires=[
        "numpy~=1.19",
        "pandas~=1.1",
        "SQLAlchemy~=1.3",
        "psycopg2~=2.8",
    ],
    python_requires="~=3.5",
    url="",
    license="LGPL-3.0",
    author="Daniel Sullivan",
    author_email="daniel.sullivan@state.mn.us",
    description="Various helpers for Postgres and Pandas",
)
