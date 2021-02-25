# coding=utf-8
from setuptools import setup

setup(
    name="postpanda_helper",
    version="0.0.3",
    packages=["postpanda_helper"],
    install_requires=[
        "numpy~=1.19",
        "pandas~=1.1",
        "SQLAlchemy~=1.3",
        "psycopg2~=2.8",
    ],
    python_requires="~=3.5",
    url="https://github.com/ds-mn/postpanda_helper",
    license="LGPL-3.0",
    author="Daniel Sullivan",
    author_email="57496265+ds-mn@users.noreply.github.com",
    description="Various helpers for Postgres and Pandas",
)
