set -e 
OUTDIR=${1:-$PWD}

if [ ! -f $OUTDIR/data/K562_RNA.tsv.gz ]
then
    echo "$OUTDIR/data/K562_RNA.tsv.gz doesn't exist, please execute the download_data.sh script"
    exit 1
fi

echo "Testing virchip-make-input-data.py script at $(date)"
echo "Expected time: 2 minutes and 40 seconds"
python virchip-make-input-data.py NRF1 $OUTDIR/data/NRF1_complete_table.tsv.gz $OUTDIR/data/ChipExpMats/NRF1\
    $OUTDIR/data/K562_RNA.tsv.gz $OUTDIR/data/RefDir --rna-cell K562 --blacklist_path\
    $OUTDIR/data/hg38_EncodeBlackListedRegions_200bpBins.bed.gz\
    --bin_size 200 --merge-chips --chromsize-path $OUTDIR/data/hg38_chrsize.tsv\
    --dnase-path $OUTDIR/data/K562_dnase.tsv.gz
echo "Finished testing virchip-make-input-data.py script at $(date)"


if [ ! -f $OUTDIR/data/NRF1_complete_table.tsv.gz ]
then
    echo "Error in making $OUTDIR/data/NRF1_complete_table.tsv.gz"
    exit 1
fi


echo "Testing virchip-predict.py script at $(date)"
echo "Expected time: 6 seconds"
python virchip-predict.py $OUTDIR/data/trainedModels $OUTDIR/data/NRF1_complete_table.tsv.gz \
    $OUTDIR/data/NRF1_predictions.tsv.gz NRF1
echo "Finished testing virchip-predict.py script at $(date)"

if [ -f $OUTDIR/data/NRF1_predictions.tsv.gz ]
then
    "Error in making $OUTDIR/data/NRF1_predictions.tsv.gz"
    exit 1
fi

echo "Testing virchip-train.py script at $(date)"
echo "Expected time: 7 minutes and 30 seconds"
TRAINDIRS=($OUTDIR/data/trainDirs/GM12878 $OUTDIR/data/trainDirs/K562)
TRAINCELLS=(GM12878 K562)
python virchip-train.py NRF1 $OUTDIR/data --test-frac 0.25 --merge-chips \
    --train-dirs ${TRAINDIRS[@]} --train-cells ${TRAINCELLS[@]} \
    --hidden-layers 2 5 --hidden-units 10 --activation-functions logistic \
    --regularization 0.001 0.01
echo "Finished testing virchip-train.py script at $(date)"


echo "Testing virchip-make-expscore-matrix.py script at $(date)"
echo "Expected time: 6 minutes and 10 seconds"
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
python virchip-make-expscore-matrix.py\
    $TF $OUTDIR $RNA chr21 --window $WINDOW\
    --qval-cutoff 4 --stringent --merge-chip\
    --num-genes $NUMGENES --chip-paths ${NPS[@]} \
    --train-cells ${CELLS[@]} --chromsize-path data/hg38_chrsize.tsv
echo "Finished testing virchip-make-expscore-matrix.py script at $(date)"
