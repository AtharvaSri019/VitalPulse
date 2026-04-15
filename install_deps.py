#!/usr/bin/env python3
"""Install required dependencies for the heart disease detection project."""

import subprocess
import sys

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")
        return False
    return True

def main():
    """Install all required packages."""
    packages = [
        "tensorflow",
        "scipy",
        "NeuroKit2",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "numpy",
        "shap",
        "PyWavelets",
        "pytest"
    ]

    print("Installing dependencies for heart disease detection project...")

    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print(f"\nFailed to install: {', '.join(failed_packages)}")
        sys.exit(1)
    else:
        print("\nAll dependencies installed successfully!")

if __name__ == "__main__":
    main()