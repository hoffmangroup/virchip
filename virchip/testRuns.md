# Virtual ChIP-seq test runs

## Downloading test datasets
```
wget https://www.pmgenomics.ca/hoffmanlab/proj/virchip/data/virchip-startup-data.tar.gz
tar -xvf virchip-startup-data.tar.gz
```


## Making input tables with virchip-make-input-data.py
```
TF=NRF1
CHROM=chr21
CELL=K562
OUTPATH=data/$TF\_$CHROM\_CompleteTable.tsv.gz
CHIPEXPDIR=data/ChipExpMats/$TF
RNAPATH=data/$CELL\_RNA.tsv.gz
REFDIR=data/RefDir
CHROMLEN=data/hg38_chrsize.tsv
BLACKLIST=data/hg38_EncodeBlackListedRegions_200bpBins.bed.gz
DNASE=data/K562_dnase.tsv.gz

python virchip-make-input-data.py\
    $TF $OUTPATH $CHIPEXPDIR $RNAPATH $REFDIR\
    --rna-cell $CELL --blacklist_path $BLACKLIST\
    --bin_size 200 --merge-chips --chromsize-path $CHROMLEN\
    --dnase-path $DNASE
```


## Predicting TF binding with virchip-predict.py
```
TF=NRF1
CHROM=chr21
CELL=K562
MODELDIR=data/trainedModels
TABLEPATH=data/$TF\_$CHROM\_CompleteTable.tsv.gz
OUTPATH=data/PredictionsIn_$CELL\_$CHROM\.tsv.gz
python virchip-predict.py $MODELDIR $TABLEPATH $OUTPATH $TF
```

## Training on a new TF with virchip-train.py
```
TF=NRF1
OUTDIR=data
TRAINDIRS=(data/trainDirs/GM12878 data/trainDirs/K562)
TRAINCELLS=(GM12878 K562)
python virchip-train.py $TF $OUTDIR --NJOBS 1 --test-frac 0.01 --merge-chips --train-dirs ${TRAINDIRS[0]} --train-cells ${TRAINCELLS[0]} --hidden-layers 5 20 --hidden-units 10 --activation-functions logistic --regularization 0.001 0.01
```

## Calculating expression score with python script
```
TF=NRF1
OUTDIR=data/ChipExpMats/NRF1-V2
mkdir $OUTDIR
RNA=data/RankOfRPKM_EncodeCCLE_RNA.tsv.gz
NPS=(data/narrowPeaks/NRF1/ENCODEProcessingPipeline_HepG2_NRF1_nan_No-Control_ENCFF313RFR.narrowpeak.gz data/narrowPeaks/NRF1/ENCODEProcessingPipeline_K562_NRF1_nan_No-Control_ENCFF161WZP.narrowpeak.gz data/narrowPeaks/NRF1/ENCODEProcessingPipeline_MCF-7_NRF1_nan_No-Control_ENCFF182QJW.narrowpeak.gz data/narrowPeaks/NRF1/GSM1462478_T47D.narrowpeak.gz data/narrowPeaks/NRF1/GSM935308_H1-hESC.narrowpeak.gz data/narrowPeaks/NRF1/GSM935309_GM12878.narrowpeak.gz data/narrowPeaks/NRF1/GSM935636_HeLa-S3.narrowpeak.gz)
CELLS=(HepG2 K562 MCF-7 T47D H1-hESC GM12878 HeLa-S3)
WINDOW=200
NUMGENES=100
python virchip-make-expscore-matrix.py $TF $OUTDIR $RNA chr21 --window $WINDOW --qval_cutoff 4 --stringent --merge_chip --num_genes $NUMGENES --chip-paths ${NPS[@]} --train-cells ${CELLS[@]} --chromsize-path data/hg38_chrsize.tsv
```

## Calculating expression score using both python script and Rscript
```
NUMGENES=5000 ## Rscript is faster and it can handle more genes
OUTDIR=data/ChipExpMats/NRF1-V3
mkdir $OUTDIR
python virchip-make-expscore-matrix.py $TF $OUTDIR $RNA chr21 --window $WINDOW --qval_cutoff 4 --stringent --merge_chip --num_genes $NUMGENES --chip-paths ${NPS[@]} --train-cells ${CELLS[@]} --chromsize-path data/hg38_chrsize.tsv --EndBeforeCor
Usage: Rscript: chip_rna_cor.R <RnaPath> <ChipMatPath> <OutPath> <Window> <NumGenes>
Rscript virchip-make-expscore-matrix.R $RNA $OUTDIR/NRF1_chr21_ChIPseqMatrix.tsv.gz $OUTDIR/NRF1_chr21_ChipExpCorrelation.tsv.gz $WINDOW $NUMGENES
```
