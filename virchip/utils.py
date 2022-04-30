import gzip
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn import neural_network
from sklearn.externals import joblib
import time


def find_best_model(comb_models, unique_chroms, dict_model):
    # Find the best model based on best MCC
    model_average_mcc = []
    for i in range(len(comb_models)):
        mcc_model_i = []
        for chrom in unique_chroms:
            mcc_temp = dict_model[chrom][i]["MCC"]
            mcc_model_i.append(mcc_temp)
        median_mcc = np.median(mcc_model_i)
        model_average_mcc.append(median_mcc)

    idx_best_hypers = np.where(
        np.array(model_average_mcc) == max(model_average_mcc))[0][0]
    hyperparam_dict = comb_models[idx_best_hypers]
    return hyperparam_dict


def get_chrom(chr_st_end):
    return chr_st_end.split(".")[0]


def find_hyperparam_neuralnet(bed_df, hidden_layers,
                              hidden_units,
                              activations, alphas,
                              learning_rates):
    # Create all combinations of hyperparameters
    comb_models = []
    for hidden_layer in hidden_layers:
        for hidden_unit in hidden_units:
            hidden_layer_sizes = tuple(
                [hidden_unit for i in range(hidden_layer)])
            for activation in activations:
                for alpha in alphas:
                    for learning_rate in learning_rates:
                        model_features = {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "activation": activation,
                            "alpha": alpha,
                            "learning_rate": learning_rate}
                        comb_models.append(model_features)

    # Cross validate by chromosomes, and record best MCC
    chroms = bed_df.index.map(get_chrom)
    unique_chroms = pd.unique(chroms)
    if len(unique_chroms) == 1:
        error_message = "Input must contain more than one chromosome " +\
            "for cross-validation based on chromosomes."
        raise ValueError(error_message)
    dict_model = {}
    for chrom in unique_chroms:
        dict_model_temp = {}
        idx_model = 0
        train_df = bed_df[chroms == chrom]
        # train_df = train_df[train_df["DNase.NarrowPeakSignal"] > 0]
        valid_df = bed_df[chroms != chrom]
        for hyperparam_dict in comb_models:
            mlp = neural_network.MLPClassifier(
                verbose=False, warm_start=True, max_iter=500,
                hidden_layer_sizes=hyperparam_dict["hidden_layer_sizes"],
                activation=hyperparam_dict["activation"],
                alpha=hyperparam_dict["alpha"],
                learning_rate=hyperparam_dict["learning_rate"])
            mlp.fit(train_df.iloc[:, train_df.columns != "Bound"],
                    np.array(train_df["Bound"]))
            prob_ar = mlp.predict_proba(
                valid_df.iloc[:, valid_df.columns != "Bound"])
            prob_df = pd.DataFrame(prob_ar)
            prob_df.columns = mlp.classes_
            mcc_scores = np.zeros(100)
            idx_cutoff = 0
            for cutoff in np.arange(0.0, 1.0, 0.01):
                pred_ar = prob_df[True] > cutoff
                mcc_score = metrics.matthews_corrcoef(
                    pred_ar, np.array(valid_df["Bound"]))
                mcc_scores[idx_cutoff] = mcc_score
                idx_cutoff = idx_cutoff + 1
            dict_model_temp[idx_model] = {"Model.ID": idx_model,
                                          "MCC": max(mcc_scores)}
            print("Validating over {}: Best MCC for model {} is {}".format(
                    chrom, idx_model, dict_model_temp[idx_model]["MCC"]))
            idx_model = idx_model + 1
        dict_model[chrom] = dict_model_temp

    # Find the best model based on best MCC
    hyperparam_dict = find_best_model(comb_models, unique_chroms, dict_model)

    mlp = neural_network.MLPClassifier(
        verbose=True, warm_start=True, max_iter=500,
        hidden_layer_sizes=hyperparam_dict["hidden_layer_sizes"],
        activation=hyperparam_dict["activation"],
        alpha=hyperparam_dict["alpha"],
        learning_rate=hyperparam_dict["learning_rate"])
    mlp.fit(bed_df.iloc[:, bed_df.columns != "Bound"],
            np.array(bed_df["Bound"]))
    column_names = [each_col for each_col in bed_df.columns
                    if each_col != "Bound"]
    dict_model = {"Features": column_names,
                  "Model": mlp}
    return dict_model


def make_accuracy_dict_multiclass(label_vec, prob_df):
    dict_summary_out = {}
    dict_posterior_out = {}
    for class_label in prob_df.columns:
        prob_vec = np.array(prob_df[class_label])
        temp_label_vec = np.array(label_vec == class_label)
        posterior_dict, summary_dict = make_accuracy_dict(
            temp_label_vec, prob_vec)
        new_post_dict = {}
        for each_key, each_item in posterior_dict.items():
            new_key = "{}.{}".format(each_key, class_label)
            new_post_dict[new_key] = each_item
        new_summary_dict = {}
        for each_key, each_item in summary_dict.items():
            new_key = "{}.{}".format(each_key, class_label)
            new_summary_dict[new_key] = each_item
        dict_summary_out.update(new_summary_dict)
        dict_posterior_out.update(new_post_dict)
    return dict_posterior_out, dict_summary_out


def make_accuracy_dict(label_vec, prob_vec):
    auc = metrics.roc_auc_score(label_vec, prob_vec)
    fpr, tpr, cutoff_roc = metrics.roc_curve(label_vec, prob_vec)
    prec, recall, cutoff_pr = metrics.precision_recall_curve(
        label_vec, prob_vec)
    aupr = metrics.auc(prec, recall, reorder=True)
    fdr = 1 - prec
    recall_dict = {}
    for fdr_cutoff in [0.05, 0.1, 0.25, 0.5]:
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


class ClassifyingObject:
    def __init__(self, clf, bed_train, bed_test, Y_train,
                 bed_df, max_rows=100000):
        self.clf = clf
        self.bed_train = bed_train
        self.Y_train = Y_train
        self.bed_df = bed_df
        bed_df_trainout = bed_df.iloc[
            np.logical_not(bed_df.index.isin(bed_train.index)), :]
        probs = parallel_benchmark(
            clf, bed_test.iloc[:, bed_test.columns != "Bound"], max_rows / 10)
        print("Calculated probability for test dataset")
        self.prob_df = pd.DataFrame(probs, columns=clf.classes_)
        probs_all = parallel_benchmark(
            clf, bed_df_trainout.iloc[:, bed_df_trainout.columns != "Bound"],
            max_rows)
        print("Calculated probability for all but trained data")
        self.prob_df_all = pd.DataFrame(probs_all, columns=clf.classes_)
        self.bed_test, self.bed_df_trainout = self.add_post_and_pred(
            bed_test, bed_df_trainout)

    def add_post_and_pred(self, bed_test, bed_df_trainout):
        # bed_test["Bound"] = self.bed_df.loc[bed_test.index, "Bound"]
        bed_test, bed_df_trainout = add_prediction(
            bed_test, bed_df_trainout, self.prob_df)
        return bed_test, bed_df_trainout

    def add_measures_to_dict(self, dict_data, rec_dict_test, var_imp):
        dict_data["Virtual ChIP-seq All"], rec_dict_all =\
            make_accuracy_dict(
                np.array(self.bed_df_trainout["Bound"]),
                self.bed_df_trainout["Class.True.Probability"])
        dict_data["Virtual ChIP-seq All auto"], rec_dict_auto =\
            make_accuracy_dict_multiclass(
                np.array(self.bed_df_trainout["Bound"]),
                self.prob_df_all)
        rec_dict = {"TestDataset": rec_dict_test,
                    "All": rec_dict_all,
                    "All.Auto": rec_dict_auto}
        out_dict = {"Posteriors": self.bed_df_trainout,
                    "Variable.Importance": var_imp,
                    "Performance": dict_data,
                    "Summary": rec_dict,
                    "Model_object": self.clf}
        return out_dict


def add_prediction(bed_test, bed_df_trainout, prob_df):
    true_label = True
    for class_label in prob_df.columns:
        new_col_name = "Class.{}.Probability".format(class_label)
        bed_test[new_col_name] = list(prob_df[class_label])
        bed_df_trainout[new_col_name] = 0
        bed_df_trainout = bed_df_trainout.set_value(
            bed_test.index, new_col_name, list(prob_df[class_label]))
    bed_test["VirtualChIP.Prediction"] = \
        prob_df[true_label] > 0.5
    bed_df_trainout[true_label] = False
    bed_df_trainout = bed_df_trainout.set_value(
        bed_test.index, "VirtualChIP.Prediction",
        bed_test["VirtualChIP.Prediction"])
    # bed_df_trainout["Bound"] = False
    # bed_df_trainout = bed_df_trainout.set_value(
    #     bed_test.index, "Bound", bed_test["Bound"])
    return bed_test, bed_df_trainout


def get_acc_idx(bed_df):
    SELECT_VARS = ["DNase.NarrowPeakSignal", "HINT",
                   "ATAC-seq.NarrowPeakSignal", "ExpScore",
                   "NarrowPeakSignal"]
    SELECT_VARS = [each for each in SELECT_VARS if each in bed_df.columns]
    list_bools = []
    for each_var in SELECT_VARS:
        bool_vec = abs(bed_df[each_var]) > 0
        list_bools.append(bool_vec)
    bool_df = pd.concat(list_bools, axis=1)
    bool_ar = bool_df.apply(sum, 1)
    ind_acc, = np.where(bool_ar)
    ind_inacc, = np.where(np.logical_not(bool_ar))
    return ind_acc, ind_inacc


def get_accessible_idx(bed_df):
    np_columns = ["DNase.NarrowPeakSignal",
                  "ATAC-seq.NarrowPeakSignal"]
    idx_narpeak, = np.where(bed_df.columns.isin(np_columns))
    if "ExpScore" in bed_df.columns:
        ind_acc, = np.where(
            bed_df.iloc[:, idx_narpeak[0]] +
            bed_df.iloc[:, idx_narpeak[-1]] +
            abs(bed_df["ExpScore"]) > 0)
    else:
        ind_acc, = np.where(
            bed_df.iloc[:, idx_narpeak[0]] +
            bed_df.iloc[:, idx_narpeak[-1]] > 0)
    return ind_acc


def sample_df(bed_df, test_frac):
    # Select chromatin accessible rows for bed_df_acc
    ind_acc = get_accessible_idx(bed_df)
    bed_df_acc = bed_df.iloc[ind_acc, :]

    # Create dataframes for training and test
    bed_train, bed_test = cross_validation.train_test_split(
        bed_df_acc, test_size=test_frac, random_state=42)
    Y_train = np.array(bed_train["Bound"])
    Y_test = np.array(bed_test["Bound"])

    # Output dataframes
    out_list = [Y_train, Y_test, bed_train, bed_test,
                bed_df_acc]
    return out_list


def make_boolean(vec):
    new_vec = vec != "U"
    return new_vec


def perf_dict_to_file(dict_data, out_link, prefix):
    out_link.write(
        "\t".join(["Experiment", "X", "Y",
                   "Cutoff", "AUC", "Dataset",
                   "Method"]) + "\n")
    for dataset in dict_data.keys():
        for acc_method in dict_data[dataset].keys():
            cur_dict = dict_data[dataset][acc_method]
            for i in range(len(cur_dict["X"]) - 1):
                out_str = "\t".join(
                    [prefix, str(cur_dict["X"][i]),
                     str(cur_dict["Y"][i]), str(cur_dict["Cutoff"][i]),
                     str(cur_dict["AUC"]),
                     dataset, acc_method])
                out_link.write(out_str + "\n")
    print("Done")


def exclude_variables(bed_df, exclude_vars):
    for ex_var in exclude_vars:
        if ex_var == "motif":
            print("Assuming motifs are {Tf}_{Motif} structure")
            print("Assuming no other columns have {}_{} format")
            print("Removing column names with motifs")
            # if a column name has underscore remove it
            ind_keep = []
            for i in range(len(bed_df.columns)):
                each_col = bed_df.columns[i]
                if "Motif" not in each_col:
                    ind_keep.append(i)
                else:
                    print("Removing {}".format(bed_df.columns[i]))
            bed_df = bed_df.iloc[:, ind_keep]
        elif ex_var == "chip":
            ind_keep = []
            for i in range(bed_df.shape[1]):
                col_name = bed_df.columns[i]
                if "Chip" in col_name or "PreviousBinding" in col_name:
                    print("Excluding {}".format(bed_df.columns[i]))
                else:
                    ind_keep.append(i)
            bed_df = bed_df.iloc[:, ind_keep]
        elif ex_var in bed_df.columns:
            if ex_var not in bed_df.columns:
                print("Failed to remove {}".format(ex_var))
            else:
                print("Removing {}".format(ex_var))
            ind_keep, = np.where(bed_df.columns != ex_var)
            bed_df = bed_df.iloc[:, ind_keep]
        else:
            print("Failed to remove {}".format(ex_var))
    return bed_df


def rename_colnames(bed_df, tf):
    col_names = []
    motif_idx = 0
    chip_idx = 0
    for col_name in bed_df.columns:
        if col_name == "NarrowPeakSignal":
            new_name = "DNase.NarrowPeakSignal"
        elif len(col_name.split("_")) == 2:
            motif_idx = motif_idx + 1
            new_name = str(col_name)
            if "JASPAR" not in new_name:
                new_name = "JASPAR." + str(col_name)
        elif len(col_name.split(".")) > 2:
            if col_name.split(".")[-1] == tf:
                chip_idx = chip_idx + 1
                new_name = "Chip" + str(chip_idx)
            else:
                new_name = col_name
        else:
            new_name = col_name
        col_names.append(new_name)
    bed_df.columns = col_names
    return bed_df


def scale_columns(bed_df):
    ind_cols, = np.where(
        np.logical_not(
            bed_df.columns.isin(
                ["Bound", "Chrom", "Start", "End"])))
    for j in ind_cols:
        max_val = max(bed_df.iloc[:, j])
        if max_val > 1:
            bed_df.iloc[:, j] = bed_df.iloc[:, j] / max_val
            print("Scaled {}".format(bed_df.columns[j]))
    return bed_df


def find_true_labels(bed_df, cell, tf, stringent):
    response_cols = []
    idx_keep = []
    idx_col = 0
    for each_col in bed_df.columns:
        col_parts = each_col.split(".")
        if col_parts[0] == cell and col_parts[-1] == tf:
            response_cols.append(each_col)
        else:
            idx_keep.append(idx_col)
        idx_col = idx_col + 1
    out_col = np.zeros(bed_df.shape[0])
    assert len(response_cols) > 0, "Reponse didn't exist"
    if stringent:
        response_df = bed_df.iloc[:, bed_df.columns.isin(response_cols)]
        if response_df.shape[1] > 1:
            bool_ar = np.array(response_df, dtype=float)
            bool_ar = bool_ar > 0
            idx_zero, = np.where(np.apply_along_axis(sum, 1, bool_ar) == 1)
            response_df.iloc[idx_zero, :] = 0
        out_col = response_df.apply(sum, 1) > 0
    else:
        for each_col in response_cols:
            idx_true, = np.where(bed_df[each_col])
            out_col[idx_true] = 1
        out_col = np.array(out_col, dtype=bool)
    return out_col, idx_keep


def average_accessible_regions(bed_df):
    np_columns = ["DNase.NarrowPeakSignal",
                  "ATAC-seq.NarrowPeakSignal"]
    if len(np.where(bed_df.columns.isin(np_columns))[0]) == 2:
        new_vals = (bed_df[np_columns[0]] +
                    bed_df[np_columns[1]]) / 2.0
        idx_keep = np.where(bed_df.columns != np_columns[-1])[0]
        bed_df = bed_df.iloc[:, idx_keep]
        bed_df[np_columns[0]] = new_vals
    elif np_columns[1] in bed_df.columns:
        bed_df[np_columns[0]] = bed_df[np_columns[1]]
        idx_keep = np.where(bed_df.columns != np_columns[-1])[0]
        bed_df = bed_df.iloc[:, idx_keep]
    return bed_df


def load_prepare_bed(bed_path, response_col, exclude_vars,
                     scale_cols=True, stringent=False,
                     merge_chips=False):
    cell, tf = ["", ""]
    if response_col != "":
        cell, tf = response_col.split(".")
    bed_df = pd.read_csv(
        bed_path, sep="\t",
        compression="gzip", index_col=0)

    # Averaging DNase-seq and ATAC-seq columns
    bed_df = average_accessible_regions(bed_df)

    # remove same cell ChIP-seq
    if cell == tf or "{}.0.{}".format(cell, tf) in bed_df.columns:
        if cell != tf:
            bed_df["Bound"], keep_idx_chip = find_true_labels(
                bed_df, cell, tf, stringent)
            keep_idx_chip.append(np.where(bed_df.columns == "Bound")[0][0])
            bed_df = bed_df.iloc[:, keep_idx_chip]
        else:
            bed_df["Bound"] = False
            print("Response variable not available, setting all to False")

        # Correct VC.Score name
        if "VC.Score" in bed_df.columns:
            bed_df["ExpScore"] = bed_df["VC.Score"]
            bed_df = bed_df.iloc[:, bed_df.columns != "VC.Score"]

        # Remove chrom,start,end columns
        # Assumes index is chrom.st.end format and will be kept
        # bed_df = bed_df.iloc[:, 3:]

        # Change name of columns
        bed_df = rename_colnames(bed_df, tf)

        # Merge all columns starting with Chip
        if merge_chips and "PreviousBinding" not in bed_df.columns:
            idx_chips, = np.where(["Chip" in each for each in bed_df.columns])
            bed_df["PreviousBinding"] = \
                bed_df.iloc[:, idx_chips].apply(sum, axis=1)
            idx_others, = \
                np.where(["Chip" not in each for each in bed_df.columns])
            bed_df = bed_df.iloc[:, idx_others]

        if len(exclude_vars) > 0:
            bed_df = exclude_variables(bed_df, exclude_vars)

        # Divide the values by maximum
        if scale_cols:
            bed_df = scale_columns(bed_df)
        return bed_df
    else:
        return []


def parallel_benchmark(clf, bed_df, step=100000):
    list_ars = []
    for i in np.arange(start=0, stop=bed_df.shape[0], step=step):
        i_end = i + step
        if i_end > bed_df.shape[0]:
            i_end = bed_df.shape[0]
        start_time = time.time()
        temp_df = bed_df.iloc[i:i_end, :]
        prob_ar = clf.predict_proba(temp_df)
        print("Calculated until {}th row".format(i_end))
        list_ars.append(prob_ar)
        end_time = time.time()
        remaining_time = (end_time - start_time) / 60
        remaining_iters = (bed_df.shape[0] - i_end) / step
        print("{} minutes to go".format(remaining_time * remaining_iters))
    prob_ar = np.concatenate(list_ars, axis=0)
    print("Concatenated posterior arrays")
    return prob_ar


def benchmark(bed_df, clf, max_rows=100000):
    dict_data = {}
    # Find accessible regions
    ind_acc = get_accessible_idx(bed_df)
    bed_df_acc = bed_df.iloc[ind_acc, :]
    Y_acc = np.array(bed_df_acc["Bound"])
    Y_all = np.array(bed_df["Bound"])

    # Calculate posterior for all of the regions
    probs_all = parallel_benchmark(
        clf, bed_df.iloc[:, bed_df.columns != "Bound"], max_rows)
    probs_all_df = pd.DataFrame(probs_all, columns=clf.classes_)
    # Benchmark posterior
    dict_data["Virtual ChIP-seq All no subsetting"], rec_dict_auto =\
        make_accuracy_dict_multiclass(Y_all, probs_all_df)

    # Calculate posterior for accessible regions
    probs = parallel_benchmark(
         clf, bed_df_acc.iloc[:, bed_df_acc.columns != "Bound"], max_rows / 10)
    prob_df = pd.DataFrame(probs, columns=clf.classes_)
    # Benchmark accessible regions
    true_label = True
    dict_data["Virtual ChIP-seq accessible"], rec_dict_acc =\
        make_accuracy_dict_multiclass(Y_acc, prob_df)

    for class_label in prob_df.columns:
        new_col_name = "Class.{}.Probability".format(class_label)
        bed_df_acc[new_col_name] = list(prob_df[class_label])
        bed_df[new_col_name] = 0
        bed_df.loc[bed_df_acc.index, new_col_name] = bed_df_acc[new_col_name]

    bed_df_acc["VirtualChIP.Prediction"] = list(prob_df[true_label] > 0.5)

    # Add data to input dataframe
    bed_df["VirtualChIP.Prediction"] = False
    bed_df.loc[bed_df_acc.index, "VirtualChIP.Prediction"] =\
        bed_df_acc["VirtualChIP.Prediction"]

    # Benchmark whole dataframe
    rec_dict = {"Accessible": rec_dict_acc,
                "All": rec_dict_auto}

    # save results to output dictionary
    out_dict = {"Posteriors": bed_df,
                "Performance": dict_data,
                "Summary": rec_dict,
                "Model_object": clf}
    return out_dict


def save_model_dict_output(out_dir, prefix, cell, tf, model_dict, save_model):
    bed_df_pos = model_dict["Posteriors"]
    dict_performance = model_dict["Performance"]
    # Save posterior probabilities and matrix
    out_path_bed = "{}/{}_{}_{}_posterior_prob.tsv.gz".format(
        out_dir, prefix, cell, tf)
    with gzip.open(out_path_bed, "wb") as out_link:
        bed_df_pos.to_csv(out_link, sep="\t")

    # Save model performance stats
    out_path_perf = "{}/{}_{}_{}_model_performance.tsv".format(
        out_dir, prefix, cell, tf)
    with open(out_path_perf, "w") as out_link:
        perf_dict_to_file(dict_performance, out_link, prefix)

    # Save summary stats
    dict_summary_stats = model_dict.get("Summary", {})
    if len(dict_summary_stats.keys()) > 0:
        out_path = "{}/{}_{}_{}_summary_stats.tsv".format(
            out_dir, prefix, cell, tf)
        with open(out_path, "w") as out_link:
            out_link.write("Validation\tStat\tValue\n")
            for name_cond, dict_stats in dict_summary_stats.items():
                for name_stat, value in dict_stats.items():
                    out_list = [name_cond, name_stat, str(value)]
                    out_link.write("\t".join(out_list) + "\n")

    # Save model
    if save_model:
        model_obj = model_dict["Model_object"]
        out_path = "{}/{}_TrainedModel.joblib.pickle".format(
            out_dir, prefix)
        print("Saving model to {}".format(out_path))
        joblib.dump(model_obj, out_path, compress=9)
