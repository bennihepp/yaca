from distutils.core import setup, Extension

ext_module = Extension('ccluster',
                   sources = ['ccluster_ext.c'])

setup(name = 'ccluster',
      version = '1.0',
      description = 'An extension implementing k-means',
      ext_modules = [ext_module])

