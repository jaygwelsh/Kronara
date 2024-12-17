# setup.py
from setuptools import setup, find_packages

setup(
    name='kronara',
    version='0.3.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy<2.0.0',
        'scikit-learn>=1.2.2',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
        'jsonschema>=4.17.3',
        'tqdm>=4.64.0',
        'joblib>=1.2.0',
        'polars>=0.17.0',
        'torch==2.2.2',
        'torchvision==0.17.2',
        'torchaudio==2.2.2',
        'hydra-core==1.3.2',
        'omegaconf==2.3.0',
        'mlflow>=1.28.0,<3.0.0',
        'captum>=0.5.0',
        'optuna>=3.0.0',
        'pytorch-lightning>=2.0.0',
        'tensorboard',
        'loguru>=0.5.3',
        'pydantic>=1.10.0,<2.0.0',
        'shap>=0.41.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.910'
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='Kronara: A production-ready ML framework.',
    url='https://github.com/yourusername/Kronara',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
