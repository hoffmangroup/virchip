from argparse import ArgumentParser
from functools import partial
import gzip
import pandas as pd
import re
from scipy import stats
import numpy as np
import os


def load_args(items=[]):
    parser = ArgumentParser(
        description="This script generates the "
        "reference expression score matrix. "
        "It calculates the Pearson correlation "
        " of gene expression with TF binding.",
        epilog='''
        Citation: Karimzadeh M. and, Hoffman MM. 2018.
        Virtual ChIP-seq: predicting transcription factor binding by
        learning from the transcriptome.
        https://doi.org/10.1101/168419.
        ''')
    requiredNamed = parser.add_argument_group('required named arguments')
    parser.add_argument(
        "tf",
        help="Name of transcription factor")
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="Directory for writing the gzipped tab-separated outputs.")
    parser.add_argument(
        "rna_path",
        metavar="rna-path",
        help="Matrix of RNA expression values for "
        "all the different cell types.")
    parser.add_argument(
        "chrom",
        help="Name of chromosome")
    requiredNamed.add_argument(
        "--chip-paths",
        metavar="CHIPS",
        nargs="*",
        help="Path to gzipped narrowPeak files of ChIP-seq data")
    requiredNamed.add_argument(
        "--train-cells",
        metavar="CELLS",
        nargs="*",
        help="Name of cell types (same order as --chip_paths)")
    requiredNamed.add_argument(
        "--chromsize-path",
        metavar="CHROMLEN",
        help="Path to 2 column chromosome and size file")
    parser.add_argument(
        "--window",
        metavar="W",
        type=int,
        default=100,
        help="Window size of the wiggle files. defaults to 100")
    parser.add_argument(
        "--qval-cutoff",
        metavar="Q",
        default=0,
        type=float,
        help="Q-value cutoff for filtering peaks. Default no filtering")
    parser.add_argument(
        "--stringent",
        action="store_true",
        help="If specified, does not consider peaks that are only in one "
        "replicate (when more than 1 replicate exists).")
    parser.add_argument(
        "--merge-chip",
        action="store_true",
        help="Specify to merge multiple ChIP-seq replicates")
    parser.add_argument(
        "--num-genes",
        metavar="NGENES",
        default=5000,
        type=int,
        help="Number of genes to use from the RNA-seq matrix. "
        "(ordered by variance among --train-cells)")
    parser.add_argument(
        "--EndBeforeCor",
        action="store_true",
        help="If specified, writes RNA-seq and ChIP-seq matrix and exits")
    args = parser.parse_args()
    for i in range(len(args.chip_paths)):
        print("Assuming {} corresponds to {}".format(
                args.chip_paths[i], args.train_cells[i]))
    if not args.merge_chip:
        print("Caution, running without merging multiple ChIP-seq replicates")
    return args


def quantileNormalize(df_input):
    # Obtained from https://github.com/ShawnLYU/Quantile_Normalize
    df = df_input.copy()
    # compute rank
    dic = {}
    for col in df:
        dic.update({col: sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis=1).tolist()
    # sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df


def subset_list(in_list, regex_list):
    out_list = []
    for item in in_list:
        ad_list = [item]
        for j in range(len(regex_list)):
            if len(ad_list) > 0:
                RE = re.search(regex_list[j], ad_list[0])
                if RE:
                    ad_list = [item]
                else:
                    ad_list = []
        if len(ad_list) > 0:
            out_list.append(ad_list[0])
    return out_list


def get_rna_df(rna_path, cells, num_genes=2000):
    rna_mat = pd.read_csv(rna_path, sep="\t",
                          compression="gzip",
                          index_col=0)
    name_genes = rna_mat.index
    rna_mat = rna_mat.iloc[:, rna_mat.columns.isin(cells)]
    variance_vec = rna_mat.apply(np.var, axis=1)
    order_var = np.argsort(variance_vec)
    genes_2k = name_genes[order_var[-num_genes:]]
    rna_mat = rna_mat.loc[rna_mat.index[order_var[-num_genes:]]]
    rna_mat.index = list(genes_2k)
    rna_df_columns = [cell_name.split(".")[0] for
                      cell_name in rna_mat.columns]
    rna_mat.columns = rna_df_columns
    rna_mat = rna_mat.reindex_axis(sorted(rna_mat.columns), axis=1)
    return rna_mat


def get_chip_ar(np_path, chrom, chromsize, window, qval_cutoff):
    window_fl = float(window)
    np_df = pd.read_csv(np_path, sep="\t",
                        header=None, compression="gzip")
    np_df = np_df[np_df.iloc[:, 8] >= qval_cutoff]
    # np_df["Ranked.qVal"] = stats.rankdata(np_df.iloc[:, 8])
    np_df["Ranked.qVal"] = np_df.iloc[:, 8]
    np_df = np_df[np_df.iloc[:, 0] == chrom]
    np_ar = np.zeros(int(round(chromsize / window_fl)))
    for i in range(np_df.shape[0]):
        st, end = np_df.iloc[i, 1:3]
        # Picking the signal value
        np_ar[int(round(st/window_fl)):(int(round(end/window_fl)) + 1)] = \
            np_df.iloc[i, -1]
    return np_ar


def make_chip_matrix(np_paths, trian_cells, tf,
                     chrom, chromsize, window,
                     merge_chip, qval_cutoff, stringent):
    tf_cell_paths = np_paths
    cell_names = train_cells
    list_dfs = []
    for cell_name in pd.unique(cell_names):
        same_cell_idxs, = np.where(np.array(cell_names) == cell_name)
        cell_id = 1
        cell_dict = {}
        for cell_idx in same_cell_idxs:
            cell_name_id = "{}.{}".format(cell_name, cell_id)
            full_path = tf_cell_paths[cell_idx]
            cell_ar = get_chip_ar(full_path, chrom, chromsize,
                                  window, qval_cutoff)
            cell_dict[cell_name_id] = cell_ar
            cell_id = cell_id + 1
        cell_df = pd.DataFrame.from_dict(cell_dict)
        if cell_df.shape[1] > 1 and stringent:
            bool_ar = np.array(cell_df, dtype=float)
            bool_ar = bool_ar > 0
            idx_zero, = np.where(np.apply_along_axis(sum, 1, bool_ar) == 1)
            cell_df.iloc[idx_zero, :] = 0
        if merge_chip:
            cell_ar = cell_df.apply(np.mean, 1)
            cell_df = pd.DataFrame(
                cell_ar, columns=["{}.{}".format(cell_name, 0)])
        list_dfs.append(cell_df)
        print("Added ChIP-seq data of {}".format(cell_name))
    tf_df = pd.concat(list_dfs, 1)
    sum_rows = tf_df.apply(sum, 1)
    tf_df = tf_df.iloc[np.where(sum_rows > 0)[0], :]
    tf_df = tf_df.reindex_axis(sorted(tf_df.columns), axis=1)
    return tf_df


def get_chrpos_dict(np_dir, window):
    out_path = "{}/Chromosomal-Index-Window-Of-{}-bp.tsv.gz".format(
        np_dir, window)
    if os.path.exists(out_path):
        dict_df = pd.read_csv(out_path, sep="\t", compression="gzip",
                              index_col=0)
    else:
        wg_path = subset_list(os.listdir(np_dir), ["wg.gz"])[0]
        wg_fullpath = "{}/{}".format(np_dir, wg_path)
        dict_list = []
        ind_genome = 0
        start = 0
        with gzip.open(wg_fullpath, "rb") as wg_link:
            for wg_line in wg_link:
                wg_line = wg_line.decode()
                if "=" in wg_line:
                    if "chrom=" in wg_line:
                        start = 0
                        new_chr = wg_line.rstrip().split("chrom=")[-1]
                        new_chr = new_chr.split(" ")[0]
                        print("Starting {} for indexing".format(new_chr))
                else:
                    # dict_ind[ind_genome] = {"chr": new_chr, "start": start,
                    #                        "end": start + window}
                    dict_list.append([ind_genome, new_chr,
                                      start, start + window])
                    start = start + window
                    ind_genome = ind_genome + 1
        # dict_df = pd.DataFrame.from_dict(dict_ind).transpose()
        dict_df = pd.DataFrame(
            dict_list, columns=["Index", "chr", "start", "end"])
        dict_df.index = dict_df["Index"]
        dict_df.drop("Index", axis=1, inplace=True)
        del dict_list
        out_link = gzip.open(out_path, "wb")
        dict_df.to_csv(out_link, sep="\t")
        out_link.close()
    return dict_df


def get_pvalue_lm(X, Y, predictions, res):
    sse = np.sum((predictions - Y) ** 2, axis=0) / float(
        X.shape[0] - X.shape[1])
    se = np.array([
        np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
        for i in range(sse.shape[0])])
    t_stat = res.coef_ / se
    pvalue = 2 * (1 - stats.t.cdf(np.abs(t_stat), Y.shape[0] - X.shape[1]))
    pvalue = pvalue[0][0]
    return pvalue


# def lm_vec_vec(vec_1, vec_2):
#     ALPHA = 0.1
#     Y = np.array([[
#         (float(each_val) - min(vec_1))/(max(vec_1) - min(vec_1))
#         for each_val in vec_1]])
#     X = np.array([[
#         (float(each_val) - min(vec_2))/(max(vec_2) - min(vec_2))
#         for each_val in vec_2]])
#     Y = Y.transpose()
#     X = X.transpose()
#     X = sm.add_constant(X[:, 0])
#     LM = sm.OLS(Y, X)
#     res = LM.fit()
#     effect_size = np.nan
#     if res.pvalues[1] < ALPHA:
#         effect_size = res.params[1]
#     return effect_size


def pearson_vec_vec(vec_1, vec_2):
    ALPHA = 0.1
    Y = np.array([
        (float(each_val) - min(vec_1))/(max(vec_1) - min(vec_1))
        for each_val in vec_1])
    X = np.array([
        (float(each_val) - min(vec_2))/(max(vec_2) - min(vec_2))
        for each_val in vec_2])
    pearson_r, pval = stats.pearsonr(X, Y)
    effect_size = np.nan
    if pval < ALPHA:
        effect_size = pearson_r
    return effect_size


def lm_vec_mat(vec_chip, rna_mat):
    # get_effsize = partial(lm_vec_vec, vec_2=vec_chip)
    get_effsize = partial(pearson_vec_vec, vec_2=vec_chip)
    cor_vals = rna_mat.apply(get_effsize, 1)
    return cor_vals


def annotate_sites(tf_df, rna_df, tf):
    estimated_time = (tf_df.shape[0] * 6) / 60.0
    print("Estimated time of {} minutes".format(estimated_time))
    get_cor = partial(lm_vec_mat, rna_mat=rna_df)
    cors_df = tf_df.apply(get_cor, 1)
    return cors_df


def write_coefs(out_dir, coef_df, job_id, other_tf, tf_id):
    coef_dir = "{}/Coefficients".format(out_dir)
    if not os.path.exists(coef_dir):
        os.makedirs(coef_dir)
    out_path = "{}/{}_{}_{}_Coefficients-Linear-Model.txt.gz".format(
        coef_dir, job_id, other_tf, tf_id)
    # out_link = gzip.open(out_path, "wb")
    coef_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    # out_link.close()
    print("Created {}".format(out_path))


def write_beds(out_dir, tf_df, tf, lab, chrpos_dict, other_tf, tf_id):
    tf_df["Index"] = tf_df.index
    out_path_all = "{}/{}_{}_{}_{}_All_Versus_"
    "OtherTFs_BindingSites.bed.gz".format(
        out_dir, lab, tf, other_tf, tf_id)
    out_link_all = gzip.open(out_path_all, "wb")
    out_link_all.write(
        "\t".join(["Chromosome", "Start", "End"] + list(tf_df.columns)) + "\n")
    # for each_label in pd.unique(tf_df["Label"]):
    #     temp_df = tf_df[tf_df["Label"] == each_label]
    #     out_path = "{}/{}_{}_{}_{}_BindingSites.bed.gz".format(
    #         out_dir, lab, TF, each_label, other_tf)
    #     out_link = gzip.open(out_path, "wb")
    for i in range(len(tf_df)):
        dict_pos = chrpos_dict.loc[tf_df.index[i]]
        ad_list = [dict_pos["chr"], str(dict_pos["start"]),
                   str(dict_pos["end"])] + list(tf_df.iloc[i, :])
        ad_str = "\t".join([str(each_val) for each_val in ad_list]) + "\n"
        # out_link.write(ad_str)
        out_link_all.write(ad_str)
        # out_link.close()
        # print("Done with {}".format(out_path))
    out_link_all.close()
    print("Done with {}".format(out_path_all))


def get_pos(idx, window=100):
    st_pos = int(idx) * window
    end_pos = (int(idx) + 1) * window
    return "{}.{}".format(st_pos, end_pos)


def complement_rna_df_with_tf(rna_df, tf_df):
    new_rna_list = []
    for each_cell in tf_df.columns:
        cell_name = ".".join(each_cell.split(".")[:-1])
        new_rna_list.append(rna_df[cell_name])
    new_rna_df = pd.concat(new_rna_list, axis=1)
    new_rna_df.columns = tf_df.columns
    return new_rna_df


def annotate_peaks_motor(out_dir, np_paths, rna_path, window,
                         tf, chrom, chromsize, train_cells,
                         merge_chip, qval_cutoff, stringent,
                         num_genes, EndBeforeCor):
    out_path = "{}/{}_{}_TfGeneExpCorrelation.tsv.gz".format(
        out_dir, tf, chrom)
    out_chip_path = "{}/{}_{}_ChIPseqMatrix.tsv.gz".format(
        out_dir, tf, chrom)
    if os.path.exists(out_path):
        raise ValueError("{} existed, exiting".format(out_path))
    if os.path.exists(out_chip_path) and EndBeforeCor:
        raise ValueError("{} existed, exiting".format(out_chip_path))
    rna_df = get_rna_df(rna_path, train_cells,
                        num_genes)
    tf_df = make_chip_matrix(
        np_paths, train_cells, tf, chrom, chromsize,
        window, merge_chip, qval_cutoff, stringent)
    print("Generated the tf_df matrix for {} {}".format(tf, chrom))
    # other_tf = all_tfs[tf_id]
    # for other_tf in pd.unique(all_tfs):
    tf_df = quantileNormalize(tf_df)
    rna_df = complement_rna_df_with_tf(rna_df, tf_df)
    rna_df = rna_df.iloc[:, rna_df.columns.isin(tf_df.columns)]
    rna_df = rna_df[rna_df.apply(sum, 1) > 0]
    if EndBeforeCor:
        out_path = "{}/{}_{}_ChIPseqMatrix.tsv.gz".format(
            out_dir, tf, chrom)
        if not os.path.exists(out_path):
            tf_df.to_csv(out_path, sep="\t", compression="gzip")
        else:
            raise ValueError("{} existed".format(out_path))
        raise ValueError("Exiting due to setting -EndBeforeCor")
    else:
        print("Analyzing {} chip".format(
            tf))
        cors_df = annotate_sites(
            tf_df, rna_df, tf)
        get_pos_partial = partial(get_pos, window=window)
        genomic_pos = chrom + "." + \
            pd.DataFrame(cors_df.index.map(get_pos_partial))
        cors_df.index = list(genomic_pos.iloc[:, 0])
        out_path = "{}/{}_{}_TfGeneExpCorrelation.tsv.gz".format(
            out_dir, tf, chrom)
        print("Created values for {}".format(out_path))
        print("Writing correlation values to {}".format(out_path))
        cors_df.to_csv(out_path, sep="\t", compression="gzip")
        print("Successfully created {}".format(out_path))


def get_chrom_dict(chromsize_path):
    out_dict = {}
    with open(chromsize_path, "r") as chromsize_link:
        for chromsize_line in chromsize_link:
            # chromsize_line = chromsize_line.decode()
            chrom, size = chromsize_line.rstrip().split("\t")
            out_dict[chrom] = int(size)
    return out_dict


if __name__ == "__main__":
    args = load_args()
    chromsize_path = args.chromsize_path
    np_paths = args.chip_paths
    train_cells = args.train_cells
    chromsize_dict = get_chrom_dict(chromsize_path)
    chrom = args.chrom
    chromsize = chromsize_dict[chrom]
    tf = args.tf
    print("Making reference matrix for {} ({} bp) of {} ChIP-seq".format(
        chrom, chromsize, tf))
    annotate_peaks_motor(args.out_dir, np_paths, args.rna_path,
                         args.window, tf, chrom, chromsize, train_cells,
                         args.merge_chip, args.qval_cutoff, args.stringent,
                         args.num_genes, args.EndBeforeCor)
