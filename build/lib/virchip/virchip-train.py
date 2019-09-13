from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib
from virchip_utils import load_prepare_bed
from virchip_utils import scale_columns
from virchip_utils import get_acc_idx
from virchip_utils import find_hyperparam_neuralnet


def get_args():
    parser = ArgumentParser(
        description="Train the Virtual ChIP-seq "
        "model on randomly selected regions from several"
        "cell types. Saves the trained model and feature "
        "names in a dictionary using joblib. "
        "--train_dirs files can be generated with virchip-make-input-data "
        "and must have one column named as "
        "<Cell>.<TF>. Columns corresponding to other "
        "ChIP-seq experiments must also follow <Cell>.<TF> format. "
        "Columns corresponding to sequence motif scores must "
        "follow JASPAR.<TfName>_<MotifID> format. "
        "Other expected columns are: ExpScore, NarrowPeakSignal, "
        "and Conservation.",
        epilog='''
        Citation: Karimzadeh M. and, Hoffman MM. 2018.
        Virtual ChIP-seq: predicting transcription factor binding by
        learning from the transcriptome.
        https://doi.org/10.1101/168419
        ''')
    requiredNamed = parser.add_argument_group('required named arguments')
    MLPoptions = parser.add_argument_group(
        'Multi-layer perceptron architecture options')
    parser.add_argument(
        "tf",
        help="Name of TF")
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="Directory for writing output files")
    requiredNamed.add_argument(
        "--train-dirs",
        metavar="TRAINING",
        required=True,
        nargs="*",
        help="Path to folders with file names as "
        "<chrom>_<tf>_<cell>_... which can be made using "
        "virchip-make-input-data. Make sure each folder "
        "excludes ChIP-seq data of the training cell type. For example, "
        "if the first folder corresponds to K562, you shouldn't "
        "use any ChIP-seq data of K562 in calculating expression score.")
    requiredNamed.add_argument(
        "--train-cells",
        metavar="CELLS",
        required=True,
        nargs="*",
        help="Name of training cell types (expects same "
        "order as --train_dirs)")
    parser.add_argument(
        "--NJOBS",
        default=1,
        type=int,
        help="Number of parallel SMP processes")
    parser.add_argument(
        "--test-frac",
        metavar="FRACTION",
        default=0.9,
        type=float,
        help="The fraction of dataset to be used for validation.")
    parser.add_argument(
        "--exclude-vars",
        metavar="EXCLUDE",
        nargs="*",
        default=[],
        help="Specify variables to exclude. Choices are exact "
        "name of the columns in --train-dirs files, "
        "or the word 'motif' (accepts multiple space "
        "separated arguments). By specifying the word 'chip', no ChIP-seq "
        "data of other cell lines will be used.")
    parser.add_argument(
        "--merge-chips",
        action="store_true",
        help="If specified, will merge ChIP-seq data of other cell types "
        "into one column. Assumes these columns are in <cell>.<tf> format.")
    MLPoptions.add_argument(
        "--hidden-layers",
        metavar="NLAYERS",
        nargs="*",
        default=[1, 2, 10],
        type=int,
        help="Space separated number of hidden layers for grid search")
    MLPoptions.add_argument(
        "--hidden-units",
        metavar="NUNITS",
        nargs="*",
        default=[10, 50, 100],
        type=int,
        help="Space separated number of hidden units for grid search")
    MLPoptions.add_argument(
        "--activation-functions",
        metavar="ACTIVATION",
        nargs="*",
        default=["logistic", "relu", "tanh"],
        help="Space separated name of activation functions for grid "
        "search. Must be supported by sickitlearn MLP")
    MLPoptions.add_argument(
        "--regularization",
        metavar="L2",
        nargs="*",
        type=float,
        default=[0.0001, 0.01],
        help="Space separated floats for regularization of MLP used "
        "for grid search")
    args = parser.parse_args()
    print("Just so you know, we're using {} cores".format(
        args.NJOBS))
    dict_train = make_file_dict(args.train_dirs, args.train_cells, args.tf)
    dict_mlp_hyper = {"hidden_layers": args.hidden_layers,
                      "hidden_units": args.hidden_units,
                      "activation": args.activation_functions,
                      "regularization": args.regularization}
    out_list = [dict_train, args.tf, args.out_dir, args.NJOBS,
                args.test_frac, args.exclude_vars,
                args.merge_chips, dict_mlp_hyper]
    return out_list


def make_file_dict(list_dirs, list_cells, tf):
    dict_files = {}
    for cell in pd.unique(list_cells):
        idx_cell = np.where(np.array(list_cells) == cell)[0][0]
        dict_files[cell] = [
            "{}/{}".format(list_dirs[idx_cell], each_path)
            for each_path in os.listdir(list_dirs[idx_cell])
            if "_{}_".format(tf) in each_path]
    return dict_files


def get_rna_dict(rna_path, cell, TFs):
    rna_df = pd.read_csv(
        rna_path, sep="\t",
        compression="gzip", index_col=0)
    rna_df = rna_df.loc[rna_df.index.isin(TFs)]
    ind_cell = np.where(rna_df.columns == cell)[0][0]
    rna_df.iloc[:, ind_cell] = stats.rankdata(rna_df.iloc[:, ind_cell])
    rna_dict = rna_df.to_dict()
    rna_dict = rna_dict[cell]
    return rna_dict


def make_accuracy_dict(label_vec, prob_vec):
    auc = metrics.roc_auc_score(label_vec, prob_vec)
    fpr, tpr, cutoff_roc = metrics.roc_curve(label_vec, prob_vec)
    prec, recall, cutoff_pr = metrics.precision_recall_curve(
        label_vec, prob_vec)
    aupr = metrics.auc(prec, recall, reorder=True)
    fdr = 1 - prec
    recall_dict = {}
    for fdr_cutoff in [0.05, 0.1, 0.255555, 0.5]:
        cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
        recall_dict["Recall at {} FDR".format(fdr_cutoff)] = \
            recall[cutoff_index]
    recall_dict["AUROC"] = auc
    recall_dict["AUPR"] = aupr
    pred_vec = np.array(prob_vec) > 0.5
    mcc_score = metrics.matthews_corrcoef(label_vec, pred_vec)
    accuracy_score = metrics.accuracy_score(label_vec, pred_vec)
    recall_dict["MCC"] = mcc_score
    recall_dict["Accuracy"] = accuracy_score
    dict_perf = {
        "ROC": {
            "X": fpr, "Y": tpr, "Cutoff": cutoff_roc, "AUC": auc},
        "PR": {
            "X": recall, "Y": prec, "Cutoff": cutoff_pr, "AUC": aupr}
        }
    return dict_perf, recall_dict


def neuralnet_classifier(bed_df, NJOBS, dict_mlp_hyper):
    print("Starting multi-layer perceptron with {} cores".format(NJOBS))
    model_dict = find_hyperparam_neuralnet(
        bed_df, hidden_layers=dict_mlp_hyper["hidden_layers"],
        hidden_units=dict_mlp_hyper["hidden_units"],
        activations=dict_mlp_hyper["activation"],
        alphas=dict_mlp_hyper["regularization"],
        learning_rates=["adaptive"])
    best_mlp = model_dict["Model"]
    return best_mlp


def train_wrapper(dict_train, tf, out_dir, NJOBS,
                  test_frac, exclude_vars, merge_chips,
                  dict_mlp_hyper):
    if len(exclude_vars) > 0:
        print("Will exclude {}".format("&".join(exclude_vars)))
    prefix = "{}_Model_TrainedOn_{}".format(tf, "_".join(dict_train.keys()))
    out_path = "{}/{}-TrainedModel.joblib.pickle".format(
        out_dir, prefix)
    if os.path.exists(out_path):
        raise ValueError("{} already exists, exiting!".format(out_path))
    list_train_dfs = []
    i = 0
    for cell, list_paths in dict_train.items():
        for each_path in list_paths:
            response_col = "{}.{}".format(cell, tf)
            bed_df = load_prepare_bed(
                each_path, response_col, exclude_vars, scale_cols=False,
                merge_chips=merge_chips)
            bed_df = bed_df.iloc[:, np.logical_not(
                                    bed_df.columns.isin(
                                        ["Chrom", "Start", "End"]))]
            # Use accessible or TF-bound regions for training
            if len(bed_df) > 0:
                ind_acc, ind_unacc = get_acc_idx(bed_df)
                bed_df_acc = bed_df.iloc[ind_acc, :]
                bed_acc_train, bed_acc_test =\
                    cross_validation.train_test_split(
                        bed_df_acc, test_size=test_frac, random_state=42)
                list_train_dfs.append(bed_acc_train)
                i = i + 1
                print(
                    "Added {} rows from {}".format(
                        bed_acc_train.shape[0], each_path))
        print("Added training data for {}".format(cell))

    # Merge data from multiple cell types into one dataframe
    bed_df_train = pd.concat(list_train_dfs)
    bed_df_train = scale_columns(bed_df_train)
    features_str = " ".join(list(bed_df_train.columns))
    print("Using the following features: {}".format(features_str))
    print("Total rows in training is {}".format(bed_df_train.shape[0]))
    clf_obj = neuralnet_classifier(
        bed_df_train, NJOBS, dict_mlp_hyper)
    print("Saving model to {}".format(out_path))
    dict_model = {"Features": list(bed_df_train.columns),
                  "Model": clf_obj}
    joblib.dump(dict_model, out_path, compress=9)


if __name__ == "__main__":
    list_args = get_args()
    dict_train, tf, out_dir, NJOBS, test_frac = list_args[:5]
    exclude_vars, merge_chips, dict_mlp_hyper = list_args[5:]
    train_wrapper(dict_train, tf, out_dir, NJOBS, test_frac,
                  exclude_vars, merge_chips, dict_mlp_hyper)
