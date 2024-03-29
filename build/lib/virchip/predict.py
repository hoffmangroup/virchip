from argparse import ArgumentParser
import os
import pandas as pd
from sklearn.externals import joblib
from utils import load_prepare_bed
from utils import get_acc_idx


def get_args():
    parser = ArgumentParser(
        description="Predict TF binding using a trained model "
        "stored in joblib format."
        "The joblib object must be a dictionary with feature names "
        "stored with key name 'Features' and the scikit-learn model"
        "stored with key name 'Model'",
        epilog='''
        Citation: Karimzadeh M. and, Hoffman MM. 2018.
        Virtual ChIP-seq: predicting transcription factor binding by
        learning from the transcriptome.
        https://doi.org/10.1101/168419
        ''')
    parser.add_argument(
        "model_dir",
        metavar="model-dir",
        help="Directory with <TF>.joblib.pickle files. Each file contains "
        "a dictionary which can be loaded using joblib, and has a "
        "'Features' key which contains list of features used for "
        "training the classifier, and 'Model' which is a "
        "scikit-learn classifier object")
    parser.add_argument(
        "table_path",
        metavar="table-path",
        help="Path to table generated by virchip-make-input-data")
    parser.add_argument(
        "out_path",
        metavar="out-path",
        help="Output file with posterior probability")
    parser.add_argument(
        "tf",
        help="Name of TF")
    args = parser.parse_args()
    list_out = [args.model_dir, args.table_path,
                args.out_path, args.tf]
    return list_out


def predict_binding(model_dir, table_path, out_path, tf):
    bed_df = load_prepare_bed(
        table_path, response_col="", exclude_vars=[],
        merge_chips=True)
    idx_ca, idx_unac = get_acc_idx(bed_df)
    bed_df = bed_df.iloc[idx_ca, :]
    out_df = bed_df.iloc[:, bed_df.columns.isin(["Chrom", "Start", "End"])]
    model_path = "{}/{}.joblib.pickle".format(model_dir, tf)
    if os.path.exists(model_path):
        model_dict = joblib.load(model_path)
        feature_names = model_dict["Features"]
        feature_names = [col_name for col_name in feature_names
                         if col_name != "Bound"]
        model_obj = model_dict["Model"]
        try:
            temp_df = bed_df[feature_names]
        except KeyError:
            "Some features don't exist"
        prob_df = model_obj.predict_proba(
            temp_df)
        prob_df = pd.DataFrame(prob_df, columns=model_obj.classes_)
        prob_df.index = bed_df.index
        out_df["Posterior.{}".format(tf)] = prob_df[True]
        print("Done with predicting {}".format(tf))
    else:
        raise ValueError(
            "{} doesn't exist, can't predict {}".format(model_path, tf))
    print("Writing {}".format(out_path))
    # with gzip.open(out_path, "wb") as out_link:
    out_df.to_csv(out_path, sep="\t", compression="gzip")


if __name__ == "__main__":
    list_args = get_args()
    model_dir, table_path, out_path, tf = list_args
    predict_binding(model_dir, table_path, out_path, tf)
