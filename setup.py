from setuptools import setup, Extension

modulenphash = Extension("nphash", 
      sources=["pynpfnv1a.cpp"])

setup(name = "nphash", 
      version = '0.1',
      description = 'Hashing numpy arrays',
      author = "Djings",
      author_email = "rogkor@gmail.com",
      license = "MIT",
      ext_modules = [modulenphash]
      )
