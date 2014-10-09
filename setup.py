from setuptools import setup, find_packages

setup(
    name='urbansim_defaults',
    version='0.1dev',
    description='Installs and runs the urbansim defaults.',
    author='Synthicity',
    author_email='support@synthicity.com',
    license='BSD',
    url='https://github.com/synthicity/urbansim_defaults',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
    ]
)
