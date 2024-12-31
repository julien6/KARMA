from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os


class InstallCommand(install):
    """Commande personnalisée pour exécuter un script bash lors de l'installation."""

    def run(self):
        # Appelle le script bash
        script_path = os.path.join(os.path.dirname(__file__), "install.sh")
        if os.path.exists(script_path):
            print("Running install_dependencies.sh...")
            subprocess.check_call(["bash", script_path])
        else:
            print(f"Script {script_path} not found. Skipping custom install.")

        # Continue avec l'installation normale
        install.run(self)


setup(
    name="karma",
    version="0.0.1",
    packages=find_packages(),
    # install_requires=requirements,
    author='John Smith',
    author_email='john.smith@academic.edu',
    description='Kubernetes Autoscaling with Resilient Multi-Agent system',
    long_description="Kubernetes Autoscaling with Resilient Multi-Agent system",
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    cmdclass={
        'install': InstallCommand,
    }
)
