from setuptools import setup


DESCRIPTION = "Virtual ChIP-seq: predicting transcription " +\
    "factor binding by learning from the transcriptome"


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='virchip',
      version='1.2.0',
      description=DESCRIPTION,
      long_description=readme(),
      classifiers=[
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 ",
        "(GPLv3)"],
      url='https://bitbucket.org/hoffmanlab/proj/virtualchipseq',
      author="Mehran Karimzadeh, Michael M. Hoffman",
      author_email='mehran.karimzadeh@uhnresearch.ca',
      license='GPLv3',
      packages=['virchip'],
      install_requires=[
          "argparse",
          "numpy",
          "pandas>=0.23.1,<0.24.0",
          "scikit-learn>=0.18.1,<0.19.0"],
      include_package_data=True,
      zip_safe=False)
