#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
VERSION ='0.0'
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("enhancer_prediction.data_preprocessing_tools.seq_to_binary2", [ "enhancer_prediction/data_preprocessing_tools/seq_to_binary2.pyx" ]),
        Extension("enhancer_prediction.post_train_tools.cython_util", [ "enhancer_prediction/post_train_tools/cython_util.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("enhancer_prediction.data_preprocessing_tools.seq_to_binary2", [ "enhancer_prediction/data_preprocessing_tools/seq_to_binary2.c" ]),
        Extension("enhancer_prediction.post_train_tools.cython_util", [ "enhancer_prediction/post_train_tools/cython_util.c" ]),
    ]

setup_info = dict(
    name='DeepShark',
    version=VERSION,
    description='Learning and predicting gene regulatory sequences in genomes',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='',
    packages=['enhancer_prediction/train','enhancer_prediction/network_constructors','enhancer_prediction/post_train_tools','enhancer_prediction/data_preprocessing_tools','enhancer_prediction/misc'],
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 0',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Operating System :: UBUNTU :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Biology/Machine learning',
    
    ],
    install_requires=['tensorflow-gpu'],
    long_description=open('README.md').read(),
)
setup(**setup_info)