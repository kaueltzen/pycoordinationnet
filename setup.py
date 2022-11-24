from setuptools import setup

setup(name='crytures',
      version='0.0.1',
      description='CRYstal feaTURES (CRYTURES)',
      long_description='file: README.md',
      license='MIT',
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      packages=['crytures'],
      install_requires=['numpy', 'pymatgen'],
      python_requires='>=3.9',
      )
