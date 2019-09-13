OUTDIR=${1:-$PWD}
wget https://www.pmgenomics.ca/hoffmanlab/proj/virchip/data/virchip-startup-data.tar.gz --no-check-certificate -O $OUTDIR
tar -xvf virchip-startup-data.tar.gz
