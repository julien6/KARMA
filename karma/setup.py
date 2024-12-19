from setuptools import setup, find_packages

setup(
    name="k8s_hpa",
    version="0.0.1",
    packages=find_packages(),
    install_requires=['gym', 'numpy', 'keras'],
    author='John Smith',
    author_email='john.smith@academic.edu',
    description='Custom cooperative environments',
    long_description="An extendable Gym for HPA in Kubernetes environments",
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
