from setuptools import setup

setup(name='percolation',
      version='0.1',
      description='Percolation',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      url='http://github.com/twesterhout/percolation',
      author='Tom Westerhout',
      author_email='t.westerhout@student.ru.nl',
      license='BSD3',
      packages=['percolation'],
      scripts=['bin/run_percolation.py'],
      package_data={
            'percolation': ['libpercolation.so'],
      },
      install_requires=[
          'click',
          'numpy',
          'psutil',
      ],
      zip_safe=False)
