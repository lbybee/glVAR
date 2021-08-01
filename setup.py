from  setuptools import setup, find_packages, Extension, dist
from Cython.Build import cythonize
import numpy as np


ext_modules=[Extension("proxcd",
                       sources=["glVAR/proxcd.pyx"],
                       include_dirs=[np.get_include()]),
            ]

setup(name='glVAR',
      version='0.1.0',
      description='A fast implementation of group-lasso with a focus on vector autoregression',
      url='https://github.com/lbybee/glVAR',
      author='Leland Bybee',
      author_email='leland.bybee@gmail.com',
      license='MIT',
      keywords=[],
      packages=['glVAR'],
      ext_modules=cythonize(ext_modules),
#      install_requires=[
#          'numpy','progressbar','numba','scipy','joblib','scikit-learn'
#      ],
      zip_safe=False)
