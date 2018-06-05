from __future__ import division
from argparse import ArgumentParser
from functools import partial
import gzip
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
import warnings


def get_chrsize_dict(chromsize_path):
    chrom_dict = {}
    with open(chromsize_path, "r") as chrom_link:
        for chrom_line in chrom_link:
            chrom, size = chrom_line.rstrip().split("\t")
            chrom_dict[chrom] = int(size)
    return chrom_dict


def get_ar_val(start, np_ar, bin_size):
    end_pos = start + bin_size + 1
    if end_pos > len(np_ar):
        end_pos = len(np_ar)
    range_vals = np.array(range(start, end_pos), dtype=int)
    out_val = np.mean(np_ar[range_vals])
    return out_val


def make_regions(chrom, chromsize, bin_size, slide_window, blacklist_path):
    st_regs = np.empty(0)
    end_regs = np.empty(0)
    for i in np.arange(0, bin_size, step=slide_window):
        st_regs_temp = np.arange(i, chromsize - bin_size, step=bin_size)
        st_regs = np.concatenate((st_regs_temp, st_regs))
        st_regs.sort()
        st_regs = np.array(st_regs, dtype=int)
        end_regs_temp = np.arange(bin_size + i, chromsize, step=bin_size)
        end_regs = np.concatenate((end_regs_temp, end_regs))
        end_regs.sort()
        end_regs = np.array(end_regs, dtype=int)
    dict_data = {"Chrom": chrom, "Start": st_regs,
                 "End": end_regs}
    if os.path.exists(blacklist_path):
        print("Filtering by blacklist regions")
        black_df = pd.read_csv(blacklist_path, sep="\t", compression="gzip")
        black_df = black_df[black_df.iloc[:, 0] == chrom]
        ind_keep = np.where(
            np.logical_not(
                pd.DataFrame(st_regs).iloc[:, 0].isin(black_df.iloc[:, 1])))[0]
        dict_data = {"Chrom": chrom, "Start": st_regs[ind_keep],
                     "End": end_regs[ind_keep]}
    else:
        print("Blacklist regions didn't exist, using all regions")
    bed_df = pd.DataFrame(dict_data)
    bed_df = bed_df[["Chrom", "Start", "End"]]
    idx_bed = bed_df["Chrom"].map(str) + "." + bed_df["Start"].map(str) + \
        "." + bed_df["End"].map(str)
    bed_df.index = list(idx_bed)
    return bed_df


def get_rna(rna_path, cell, gene_names):
    rna_df = pd.read_csv(
        rna_path, sep="\t", index_col=0,
        compression="gzip")
    rna_df = rna_df[rna_df.index.isin(gene_names)]
    same_cell = "NA"
    for each_col in rna_df.columns:
        cell_part = "_".join(each_col.split("_")[:2])
        if cell == cell_part:
            same_cell = each_col
    if same_cell == "NA":
        raise ValueError("Cell {} didn't exist".format(cell))
    rna_df["Rank"] = stats.rankdata(rna_df[same_cell])
    rna_dict = rna_df["Rank"].to_dict()
    return rna_dict


def get_cca(chip_cors, rna_vec):
    Y_vec = np.array(
        [[each_val/max(chip_cors) for each_val in chip_cors]])
    X_vec = np.array(
        [[each_val/max(rna_vec) for each_val in rna_vec]])
    Y_vec = Y_vec.transpose()
    X_vec = X_vec.transpose()
    cca_obj = CCA(n_components=1)
    cca_obj.fit(X_vec, Y_vec)
    r_squared_canonical = cca_obj.score(X_vec, Y_vec)
    return r_squared_canonical


def get_spearman(chip_cors, rna_vec):
    if len(chip_cors) - sum(np.isnan(chip_cors)) < 4:
        rho, pval = 0, 1
    else:
        rho, pval = spearmanr(rna_vec, chip_cors, nan_policy="omit")
    return rho


def get_start_end(chrom_parts):
    e_parts = [each for each in chrom_parts if "e" in each]
    idx_es = np.where(["e" in each for each in chrom_parts])[0]
    poses = []
    for i in range(len(e_parts)):
        e_part = e_parts[i]
        idx_e = idx_es[i]
        new_num = int(
            float(
                "{}.{}".format(chrom_parts[idx_e - 1], e_part)))
        poses.append(new_num)
    if len(poses) == 2:
        st, end = poses
    elif len(poses) == 1:
        if idx_es[0] == 2:
            st = poses[0]
            end = int(chrom_parts[-1])
        else:
            st = int(chrom_parts[1])
            end = poses[0]
    return st, end


def make_array(vec_val, chrom_pos, chromsize):
    out_ar = np.zeros(chromsize)
    for i in range(len(chrom_pos)):
        chr_pos = chrom_pos[i]
        chrom_parts = chr_pos.split(".")
        if len(chrom_parts) > 3:
            st, end = get_start_end(chrom_parts)
        else:
            st = int(float(chrom_parts[1]))
            end = int(float(chrom_parts[2]))
        out_ar[(st - 1):end] = vec_val[i]
    return out_ar


def add_vc(bed_df, chrom, chipexp_dir, tf,
           rna_path, bin_size, chromsize, rna_cell):
    vc_path = "{}/{}_{}_TfGeneExpCorrelation.tsv.gz".format(
        chipexp_dir, tf, chrom)
    vc_map = pd.read_csv(vc_path, compression="gzip",
                         sep="\t", index_col=0)
    rna_dict = get_rna(rna_path, rna_cell, vc_map.columns)
    gene_names = rna_dict.keys()
    gene_names.sort()
    vc_map = vc_map.iloc[:, vc_map.columns.isin(gene_names)]
    vc_map = vc_map.reindex_axis(sorted(vc_map.columns), axis=1)
    rna_vec = np.array([rna_dict[gene_name] for gene_name in gene_names])
    get_cca_motor = partial(get_spearman, rna_vec=rna_vec)
    cor_vec = np.apply_along_axis(get_cca_motor, axis=1, arr=np.array(vc_map))
    cor_ar = make_array(cor_vec, vc_map.index, chromsize)
    get_cor_motor = partial(
        get_ar_val, np_ar=cor_ar, bin_size=bin_size)
    bed_df["ExpScore"] = bed_df.iloc[:, 1].map(get_cor_motor)
    return bed_df


def find_overlap(start, cream_ar, bin_size, return_bool=True):
    out_val = np.mean(cream_ar[start:(start + bin_size + 1)])
    if return_bool and out_val > 0:
        out_val = 1
    return out_val


def get_narpeak_ar(np_df, chromsize):
    np_ar = np.zeros(chromsize, dtype=float)
    for i in range(np_df.shape[0]):
        ind_regs = range(np_df.iloc[i, 1], (np_df.iloc[i, 2] + 1))
        np_ar[ind_regs] = np_df["NarrowPeakSignal"].iloc[i]
    return np_ar


def get_motif_paths(ref_dir, tf, chrom):
    chrom_files = [each_path for each_path in os.listdir(ref_dir)
                   if each_path.split("_")[0] == chrom and
                   each_path.split("_")[1] == "FIMO"]
    motif_paths = []
    for chrom_file in chrom_files:
        tf_part = chrom_file.split("_")[3]
        if tf.lower()[:3] in tf_part.lower():
            motif_paths.append("{}/{}".format(ref_dir, chrom_file))
    return motif_paths


def add_chromatin_acc(bed_df, dnase_path, chrom, chromsize, bin_size):
    narrowpeak_columns = ["Chrom", "Start", "End", "Name", "Score",
                          "Strand", "NarrowPeakSignal",
                          "pVal", "qVal", "Summit"]
    np_df = pd.read_csv(dnase_path, sep="\t", compression="gzip",
                        header=None)
    np_df.columns = narrowpeak_columns
    np_df = np_df[np_df.iloc[:, 0] == chrom]
    np_ar = get_narpeak_ar(np_df, chromsize)
    find_overlap_motor = partial(
        find_overlap, cream_ar=np_ar, bin_size=bin_size,
        return_bool=False)
    bed_df["NarrowPeakSignal"] = bed_df.iloc[:, 1].map(find_overlap_motor)
    return bed_df


def idx_to_bed(bed_df, chrom):
    num_cols = bed_df.shape[1]

    def get_start(chr_st_end):
        return(int(chr_st_end.split(".")[1]))
    starts = bed_df.index.map(get_start)

    def get_end(chr_st_end):
        return(int(chr_st_end.split(".")[2]))
    ends = bed_df.index.map(get_end)
    bed_df["Chrom"] = chrom
    bed_df["Start"] = starts
    bed_df["End"] = ends
    bed_df = bed_df.iloc[:, [-3, -2, -1] + range(num_cols)]
    return bed_df


def get_args():
    parser = ArgumentParser(
        description="Generate a table of Cistrome and ENCODE "
        "ChIP-seq data, sequence motif scores, genomic conservation, "
        "expression score, and chromatin accessibility data "
        "for the transcription factor of interest.",
        epilog='''
        Citation: Karimzadeh M. and, Hoffman MM. 2018.
        Virtual ChIP-seq: predicting transcription factor binding by
        learning from the transcriptome.
        https://doi.org/10.1101/168419
        ''')
    requiredNamed = parser.add_argument_group('required named arguments')
    parser.add_argument(
        "tf",
        help="Name of transcription factor")
    parser.add_argument(
        "out_path",
        metavar="out-path",
        help="Output file")
    parser.add_argument(
        "chipexp_dir",
        metavar="chipexp-dir",
        help="Directory with tables named as as "
        "<TF>_<chrom>_TfGeneExpCorrelation.tsv.gz which "
        "store correlation of gene expression and ChIP-seq "
        "binding at each genomic bin."
        "Execute virchip-download.sh and specify data/chipExpDir")
    parser.add_argument(
        "rna_path",
        metavar="rna-path",
        help="Matrix of gene expression values with first column "
        "as gene symbols and other columns as cell types")
    parser.add_argument(
        "ref_dir",
        metavar="ref-dir",
        help="Directory with <chrom>_<tf>_ChIPseqdata.tsv.gz files, "
        "which contain publicly available ChIP-seq data,"
        "<chrom>_FIMO_JASPAR_<MotifName>_<MotifID>.tsv.gz files "
        "which contain sequence "
        "motif scores, and <chrom>_PhastCons.tsv.gz files which "
        "contain genomic conservation scores. "
        "Execute virchip-download.sh and specify data/RefTables")
    parser.add_argument(
        "--rna-cell",
        metavar="CELL",
        default="parallel",
        help="Column name in rna_path matrix corresponding to "
        "cell of interest. If not specified, assumes --dnase_path "
        "is a directory with dnase files where rna_cell is <cell>_.. "
        "and out_path is directory for writing output "
        "file named as <cell>_<tf>_<accession>_virtual_"
        "chip_complete_table.tsv.gz. It will use --array_id "
        "to select cell types using files in --dnase_path")
    parser.add_argument(
        "--blacklist_path",
        metavar="BLACKLIST",
        default="0",
        help="Blacklist regions. Must be "
        "binned into same length as --bin_size")
    requiredNamed.add_argument(
        "--chromsize-path",
        metavar="CHROMLEN",
        required=True,
        help="File with chromosome sizes in 2-column format")
    requiredNamed.add_argument(
        "--dnase-path",
        metavar="DNASE",
        required=True,
        help="Chromatin accessibility narrow peak file")
    parser.add_argument(
        "--bin_size",
        default=200,
        type=int,
        metavar="BP",
        help="Genomic bin size")
    parser.add_argument(
        "--array-id",
        default="SGE_TASK_ID",
        metavar="JOBARRAY",
        help="If --rna_cell is not specified, "
        "it will use the specified environmental variable "
        "to select one of the files in --dnase_path")
    parser.add_argument(
        "--merge-chips",
        action="store_true",
        help="The table will contain one column for each cell type "
        "with ChIP-seq of transcription factor (from --ref-dir)."
        "Specify --merge-chips to replace these columns "
        "with one column named 'PreviousBinding' which is the row sum "
        "of all publicly available ChIP-seq data.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    dnase_path = args.dnase_path
    rna_cell = args.rna_cell
    parallel = False
    if rna_cell == "parallel":
        parallel = True
    out_path = args.out_path
    if parallel:
        try:
            job_id = int(os.environ[args.array_id]) - 1
        except:
            raise ValueError(
                "Specify --array_id properly, or alternatively "
                "use --rna_cell.")
        dnase_paths = [each for each in os.listdir(dnase_path)
                       if ".narrowpeak.gz" in each]
        dnase_paths.sort()
        dnase_path = dnase_paths[job_id]
        tissue = dnase_path.split("_")[1]
        age = dnase_path.split("_")[5]
        acc_id = dnase_path.split("_")[-1].split(".narrowpeak.gz")[0]
        rna_cell = "{}_{}".format(tissue, age)
        dnase_path = "{}/{}".format(args.dnase_path, dnase_path)
        out_cell = "{}/{}".format(args.out_path, rna_cell)
        if not os.path.exists(out_cell):
            os.makedirs(out_cell)
        out_path = "{}/{}_{}_{}_virtual_chip_complete_table.tsv.gz".format(
            out_cell, rna_cell, args.tf, acc_id)
        print("Outpath set to {}, dnase_path set to {}".format(
                out_path, dnase_path))
    if os.path.exists(out_path):
        raise ValueError("File exists {}".format(out_path))
    chromsize_dict = get_chrsize_dict(args.chromsize_path)
    chroms = chromsize_dict.keys()
    chroms.sort()
    list_dfs = [[] for chrom in chroms]
    i = 0
    for chrom in chroms:
        chromsize = chromsize_dict[chrom]
        cons_path = "{}/{}_PhastCons.tsv.gz".format(
            args.ref_dir, chrom)
        CONS_EXISTS = os.path.exists(cons_path)
        tf_path = "{}/{}_{}_ChIPseqdata.tsv.gz".format(
            args.ref_dir, chrom, args.tf)
        CHIP_EXISTS = os.path.exists(tf_path)
        if CONS_EXISTS and CHIP_EXISTS:
            motif_paths = get_motif_paths(args.ref_dir, args.tf, chrom)
            bed_df = make_regions(chrom, chromsize, args.bin_size,
                                  50, args.blacklist_path)
            cons_df = pd.read_csv(cons_path, sep="\t", index_col=0,
                                  compression="gzip")
            bed_df[cons_df.columns[0]] = 0
            bed_df.loc[cons_df.index, cons_df.columns[0]] = cons_df.iloc[:, 0]
            print("Loaded genomic conservation info for {} of {}".format(
                    chrom, args.tf))
            # bed_df = idx_to_bed(bed_df, chrom)
            bed_df = add_vc(bed_df, chrom, args.chipexp_dir,
                            args.tf, args.rna_path, args.bin_size,
                            chromsize, rna_cell)
            print("Added expression score for {} of {}".format(
                    chrom, args.tf))
            chip_df = pd.read_csv(tf_path, sep="\t", index_col=0,
                                  compression="gzip")
            if args.merge_chips:
                bed_df["PreviousBinding"] = 0
                bed_df.loc[chip_df.index, "PreviousBinding"] = \
                    chip_df.apply(sum, axis=1)
            else:
                for each_col in chip_df.columns:
                    bed_df[each_col] = 0
                    bed_df.loc[chip_df.index, each_col] = chip_df[each_col]
            print("Added ChIP-seq data for {} of {}".format(
                    chrom, args.tf))
            for motif_path in motif_paths:
                motif_df = pd.read_csv(motif_path, sep="\t",
                                       index_col=0, compression="gzip")
                bed_df[motif_df.columns[0]] = 0
                bed_df.loc[motif_df.index, motif_df.columns[0]] = \
                    motif_df[motif_df.columns[0]]
                print("Added motif {} for {} of {}".format(
                        motif_df.columns[0], chrom, args.tf))
            bed_df = add_chromatin_acc(bed_df, dnase_path, chrom,
                                       chromsize, args.bin_size)
            print("Added chromatin accessibility data for {} of {}".format(
                    chrom, args.tf))
            bed_df = bed_df.iloc[
                np.where(bed_df.iloc[:, 4:].apply(sum, axis=1) != 0)[0], :]
            list_dfs[i] = bed_df
            print("Done with {}".format(chrom))
            i = i + 1
            del bed_df
        else:
            chrom_failed_message = "Data for {} didn't exist".format(chrom)
            warnings.warn(chrom_failed_message)
    print("Done with all chromosomes")
    list_dfs = [each_df for each_df in list_dfs
                if len(each_df) > 0]
    bed_df = pd.concat(list_dfs, axis=0)
    with gzip.open(out_path, "wb") as out_link:
        bed_df.to_csv(out_link, sep="\t")
        print("Successfully created {}".format(args.out_path))
