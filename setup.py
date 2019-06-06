#from distutils.core import setup
from setuptools import setup, find_packages
from distutils.extension import Extension
import re
import os
import codecs
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

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
        Extension("deepgmap.data_preprocessing_tools.seq_to_binary2", [ "deepgmap/data_preprocessing_tools/seq_to_binary2.pyx" ]),
        #Extension("data_preprocessing_tools.queue", [ "deepgmap/data_preprocessing_tools/queue.pyx" ],libraries=["calg"]),
        
        Extension("deepgmap.post_train_tools.cython_util", [ "deepgmap/post_train_tools/cython_util.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("deepgmap.data_preprocessing_tools.seq_to_binary2", [ "deepgmap/data_preprocessing_tools/seq_to_binary2.c" ]),
        Extension("deepgmap.post_train_tools.cython_util", [ "deepgmap/post_train_tools/cython_util.c" ]),
    ]
#print(find_version("deepgmap", "__init__.py"))
setup(
    name='DeepGMAP',
    #version=VERSION,
    version=find_version("deepgmap", "__init__.py"),
    description='Learning and predicting gene regulatory sequences in genomes',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='',
    packages=['deepgmap','deepgmap.train','deepgmap.network_constructors','deepgmap.post_train_tools','deepgmap.data_preprocessing_tools','deepgmap.misc'],
    #packages=find_packages('deepgmap'),
    #packages=['deepgmap.'],
    package_dir={'DeepGMAP':'deepgmap'},
    #package_data = {
    #     '': ['enhancer_prediction/*', '*.pyx', '*.pxd', '*.c', '*.h'],
    #},
    scripts=['bin/deepgmap',
                   ],
    #packages=find_packages(),
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License ',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        
    
    ],
    install_requires=['tensorflow-gpu', 'numpy', 'matplotlib', 'sklearn', 'tornado', 'natsort', 'psutil', 'pyBigWig'],
    long_description=open('README.rst').read(),
)

