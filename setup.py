#from distutils.core import setup
from setuptools import setup, find_packages
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
        Extension("data_preprocessing_tools.seq_to_binary2", [ "deepgmap/data_preprocessing_tools/seq_to_binary2.pyx" ]),
        #Extension("data_preprocessing_tools.queue", [ "deepgmap/data_preprocessing_tools/queue.pyx" ],libraries=["calg"]),
        
        Extension("post_train_tools.cython_util", [ "deepgmap/post_train_tools/cython_util.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("data_preprocessing_tools.seq_to_binary2", [ "deepgmap/data_preprocessing_tools/seq_to_binary2.c" ]),
        Extension("post_train_tools.cython_util", [ "deepgmap/post_train_tools/cython_util.c" ]),
    ]

setup(
    name='DeepShark',
    version=VERSION,
    description='Learning and predicting gene regulatory sequences in genomes',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='',
    #package_dir={'enhancer_prediction' : 'enhancer_prediction'},
    #packages=['enhancer_prediction','enhancer_prediction.train','enhancer_prediction.network_constructors','enhancer_prediction.post_train_tools','enhancer_prediction.data_preprocessing_tools','enhancer_prediction.misc'],
    packages=find_packages('deepgmap'),
    
    package_dir={'':'deepgmap'},
    #package_data = {
    #     '': ['enhancer_prediction/*', '*.pyx', '*.pxd', '*.c', '*.h'],
    #},
    scripts=['bin/deepgmap',
                   ],
    #packages=find_packages(),
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
    install_requires=['tensorflow-gpu', 'numpy', 'matplotlib'],
    long_description=open('README.md').read(),
)


