from setuptools import setup, find_packages

setup(
    name='urbansim_defaults',
    version='0.2.dev2',
    description='Installs and runs the urbansim defaults.',
    author='UrbanSim, Inc.',
    author_email='info@urbansim.com',
    license='BSD',
    url='https://github.com/udst/urbansim_defaults',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests'])
)
