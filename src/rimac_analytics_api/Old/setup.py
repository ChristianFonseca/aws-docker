from setuptools import setup, find_packages

#from distutils.core import setup


setup(
    name='rimac-analytics',
    version='0.5.0',
    description='Rimac Analytics API',
    author='Diego y Ricardito',
    author_email='ddce.2005@gmail.com',
    zip_safe=False,
    url='https://bitbucket.org/rimac-analytics/rimac-analytics-api',
    license='MIT',
    #packages=['rimac_analytics'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3'
)
