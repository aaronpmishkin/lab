import setuptools  # type: ignore

setuptools.setup()

# import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

# setuptools.setup(
#     name="lab",
#     version="0.1.0",
#     author="Aaron Mishkin",
#     author_email="amishkin@cs.stanford.edu",
#     description="Linear Algebra Backends: a lightweight interface for linear algebra engines.",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/aaronpmishkin/lab",
#     project_urls={
#         "Bug Tracker": "https://github.com/aaronpmishkin/lab/issues",
#     },
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     zip_safe=False,
#     package_dir={"": "src"},
#     package_data={"lab": ["py.typed"]},
#     packages=setuptools.find_packages(where="src"),
#     python_requires=">=3.6",
# )
