

Virtual ChIP-seq: predicting TF binding by learning from the transcriptome
==========================================================================


Introduction
------------

**The free Virtual ChIP-seq software package predicts binding of many transcription factors
in any cell type with available data on chromatin accessibility and gene expression.**


Virtual ChIP-seq is a supervised *binary classification* multi-layer perceptron wrapper.
You can re-train the MLP by trying different parameters or adding other features.
You can also use the already trained models *(one model per TF)* to predict binding of
certain TFs in a new cell type with only *RNA-seq data* and *chromatin accessibility data*.



Required data
-------------

Virtual ChIP-seq uses several genomic information such as DNA conservation
and transcription factor sequence preference. Reference matrices with
this information are already created for the GRCh38 genome. Please note that
TF sequence motif scores are only calculated for any region that has overlap
with any of the Roadmap consortium DNase-seq peaks. We used JASPAR 2016 motif
database.

Virtual ChIP-seq also uses epigenomic information, such as previous data
on binding of TFs at each genomic position. These are also pre-calculated in
reference matrices which can be downloaded from Zenodo_.
The *virchip-make-input-data.py* script merges these reference
matrices which include Cistrome DB, ENCODE, PhastCons genomic conservation, and outputs of
fimo on all JASPAR DB motifs for any genomic region
which was accessible in any of the Roadmap tissue DNase-seq data.


If you want to create a new expression score reference matrix (e.g. for a new TF)
you can use the *virchip-make-expscore-matrix.py* script.
This script calculates all pairwise Pearson correlation R and is slow (around 18 CPU hours).
You can use the **-EndBeforeCor** option and run the output with *virchip-make-expscore-matrix.R*.
This Rscript bootstraps samples to estimate which Pearson R correlation cutoff is associated
with the p-value cutoff for masking correlations which are not significant.



.. _Zenodo: https://doi.org/10.5281/zenodo.823297

Cell type specific data
==================

Use the *virchip-make-input-data.py* script to create a reference table
with predictive features of TF binding that Virtual ChIP-seq uses for training and prediction.
You need a gzipped RNA-seq matrix where rows are Hugo gene symbols and
columns are different cell types (at least one cell). You also need a standard gzipped
narrow peak file with chromatin accessibility information on your cell type::

    wget https://www.pmgenomics.ca/hoffmanlab/proj/virchip/data/virchip-startup-data.tar.gz 
    tar -xvf virchip-startup-data.tar.gz
    python virchip-make-input-data.py NRF1 data/NRF1_complete_table.tsv.gz data/ChipExpMats/NRF1\
        data/K562_RNA.tsv.gz data/RefDir --rna-cell K562 --blacklist_path\
        data/hg38_EncodeBlackListedRegions_200bpBins.bed.gz\
        --bin_size 200 --merge-chips --chromsize-path data/hg38_chrsize.tsv\
        --dnase-path K562_dnase.tsv.gz


Prediction
----------

We have concatenated data of several cell types and learnied parameters of multi-layer perceptron
for each TF. These models are available from Virtual ChIP-seq datasets deposited at Zenodo_.
You can use virchip-predict.py and predict binding of 70 TFs (36 with MCC > 0.3).
*virchip-predict.py* assumes that a directory contains trained models named as <TF>.joblib.pickle::

    python virchip-predict.py data/trainedModels data/NRF1_complete_table.tsv.gz \
        data/NRF1_predictions.tsv.gz NRF1


.. _Zenodo: https://doi.org/10.5281/zenodo.823297



Training
--------

If you want to train new data with Virtual ChIP-seq, make sure to exclude training cell type
information from ChIP-seq data and matrix for association of gene expression and TF binding.
Otherwise your model would overfit. The *virchip-train.py* script expects a repository where 
subfolders are different cell types, and within each subfolder there are per chromosome files
with necessary information for prediction of each TF. It will rename the column <Cell>.0.<TF> to
"Bound", and will optimize hyper-parameters for the multi-layer perceptron.
If you specify multiple training cells, it will concatenate the data of these cells prior to learning,
so the learned model would be more generalizable to new cell types::

    TRAINDIRS=(data/trainDirs/GM12878 data/trainDirs/K562)
    TRAINCELLS=(GM12878 K562)
    python virchip-train.py NRF1 $OUTDIR --test-frac 0.01 --merge-chips \
        --train-dirs ${TRAINDIRS[@]} --train-cells ${TRAINCELLS[@]} \
        --hidden-layers 5 20 --hidden-units 10 --activation-functions logistic \
        --regularization 0.001 0.01


Expression score
----------------

We have provided references matrices for calculating the expression score in a new cell type.
If you want to generate a new reference matrix (e.g. for a new TF), you can do that
using the stand-alone python script *virchip-make-expscore-matrix.py*::

    TF=NRF1
    OUTDIR=data/ChipExpMats/NRF1-V2
    mkdir $OUTDIR
    RNA=data/RankOfRPKM_EncodeCCLE_RNA.tsv.gz
    NPS=(data/narrowPeaks/NRF1/ENCODEProcessingPipeline_HepG2_NRF1_nan_No-Control_ENCFF313RFR.narrowpeak.gz
         data/narrowPeaks/NRF1/ENCODEProcessingPipeline_K562_NRF1_nan_No-Control_ENCFF161WZP.narrowpeak.gz
         data/narrowPeaks/NRF1/ENCODEProcessingPipeline_MCF-7_NRF1_nan_No-Control_ENCFF182QJW.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM1462478_T47D.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935308_H1-hESC.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935309_GM12878.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935636_HeLa-S3.narrowpeak.gz)
    CELLS=(HepG2 K562 MCF-7 T47D H1-hESC GM12878 HeLa-S3)
    WINDOW=200
    NUMGENES=100
    python virchip-make-expscore-matrix.py \
        $TF $OUTDIR $RNA chr21 --window $WINDOW \
        --qval_cutoff 4 --stringent --merge_chip \
        --num_genes $NUMGENES --chip-paths ${NPS[@]} \
        --train-cells ${CELLS[@]} --chromsize-path data/hg38_chrsize.tsv


This script performs vectorized iterations between every pair of genomic region (in ChIP-seq data)
and gene (in RNA-seq data). Since R has a more efficient build of the Pearson correlation matrix,
you can combined this script with *virchip-make-expscore-matrix.py*.
To do this, please specify the **--EndBeforeCor** option and run the Rscript similar
to the example below::

    TF=NRF1
    OUTDIR=data/ChipExpMats/NRF1-V2
    mkdir $OUTDIR
    RNA=data/RankOfRPKM_EncodeCCLE_RNA.tsv.gz
    NPS=(data/narrowPeaks/NRF1/ENCODEProcessingPipeline_HepG2_NRF1_nan_No-Control_ENCFF313RFR.narrowpeak.gz
         data/narrowPeaks/NRF1/ENCODEProcessingPipeline_K562_NRF1_nan_No-Control_ENCFF161WZP.narrowpeak.gz
         data/narrowPeaks/NRF1/ENCODEProcessingPipeline_MCF-7_NRF1_nan_No-Control_ENCFF182QJW.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM1462478_T47D.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935308_H1-hESC.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935309_GM12878.narrowpeak.gz
         data/narrowPeaks/NRF1/GSM935636_HeLa-S3.narrowpeak.gz)
    CELLS=(HepG2 K562 MCF-7 T47D H1-hESC GM12878 HeLa-S3)
    WINDOW=200
    NUMGENES=100
    python virchip-make-expscore-matrix.py $TF $OUTDIR $RNA chr21 \
        --window $WINDOW --qval_cutoff 4 --stringent --merge_chip  \
        --num_genes $NUMGENES --chip-paths ${NPS[@]} --train-cells  \
        ${CELLS[@]} --chromsize-path data/hg38_chrsize.tsv




Calculating expression score using both python script and Rscript
-----------------------------------------------------------------


Example code::

    NUMGENES=5000 ## Rscript is faster and it can handle more genes
    OUTDIR=data/ChipExpMats/NRF1-V3
    mkdir $OUTDIR
    python virchip-make-expscore-matrix.py $TF $OUTDIR $RNA chr21 \
        --window $WINDOW --qval_cutoff 4 --stringent --merge_chip \
        --num_genes $NUMGENES --chip-paths ${NPS[@]} --train-cells ${CELLS[@]} \
        --chromsize-path data/hg38_chrsize.tsv --EndBeforeCor
    # Usage: Rscript: chip_rna_cor.R <RnaPath> <ChipMatPath> <OutPath> <Window> <NumGenes>
    Rscript virchip-make-expscore-matrix.R $RNA $OUTDIR/NRF1_chr21_ChIPseqMatrix.tsv.gz $OUTDIR/NRF1_chr21_ChipExpCorrelation.tsv.gz $WINDOW $NUMGENES




Quick start
-----------

We have tested Virtual ChIP-seq installation on a CentOS system using python 2.7.11.
Virtual ChIP-seq requires numpy and pandas and it uses other python modules such as:


* Numpy
* Pandas
* scikit-learn (v0.18.1)


If you want to use pre-trained Virtual ChIP-seq models, newer versions of scikit-learn don't work.
Scikit-learn changed their data structure for saving the model parameters and they don't
have an API for extracting and re-saving the model parameters.
At this point, unfortunately, you cannot use pre-trained Virtual ChIP-seq models with newer versions.
Virtual ChIP-seq uses mercurial version control. Make sure that mercurial (hg) is installed.
Download Virtual ChIP-seq to the directory of your python packages using::

    hg clone https://bitbucket.org/hoffmanlab/virchip
    cd virchip
    python setup.py install


Downloading Virtual ChIP-seq supplementary data from Zenodo takes a lot of time.
Here we show one example with a subset of data for chr21 of NRF1::

    wget https://www.pmgenomics.ca/hoffmanlab/proj/virchip/data/virchip-startup-data.tar.gz
    tar -xvf virchip-startup-data.tar.gz


First we generate the a table with required features::

   python virchip-make-input-data.py NRF1 data/NRF1_complete_table.tsv.gz data/ChipExpMats/NRF1\
        data/K562_RNA.tsv.gz data/RefDir --rna-cell K562 --blacklist_path\
        data/hg38_EncodeBlackListedRegions_200bpBins.bed.gz\
        --bin_size 200 --merge-chips --chromsize-path data/hg38_chrsize.tsv\
        --dnase-path K562_dnase.tsv.gz


Now we will predict binding of NRF1 using an RNA-seq table and a reference matrix located at virchip/data::

    python virchip-predict.py data/trainedModels data/NRF1_complete_table.tsv.gz\
        data/NRF1_predictions.tsv.gz NRF1



Contact, support and questions
------------------------------

For support of Umap, please user our `mailing list <https://groups.google.com/forum/#!forum/virtual-chip-seq>`_.
Specifically, if you want to report a bug or request a feature,
please do so using
the `Virtual ChIP-seq issue tracker <https://bitbucket.org/hoffmanlab/virtualchipseq/issues>`_.
We are interested in all comments on the package,
and the ease of use of installation and documentation.


Credits
-------


This package is written and maintained by Mehran Karimzadeh, under supervision of Dr. Michael M. Hoffman.
