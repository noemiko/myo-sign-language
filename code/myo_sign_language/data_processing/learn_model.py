from sklearn.metrics import classification_report, confusion_matrix
import sklearn.ensemble
from sklearn import metrics
from sklearn import svm
from scipy.fftpack import fft
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
from scipy.fftpack import fft
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.datasets import fetch_california_housing
from data_processing import process_from_files

import itertools
from scipy.signal import butter, lfilter, medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier

TESTING_DATA_PERCENT = 0.3
TEACHING_DATA_PERCENT = 0.7
ROWS_IN_TIME_WINDOW = 50
N_CLUSTERS = 10  # 9, 14, 17, 18, 22



SENSORS_NAMES = ["acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3",
                 "orientation1", "orientation2", "orientation3", "orientation4",
                 "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7", "emg8"]


def test_filter_on_sensors_data(array_with_columns_as_row, filter):
    for row in array_with_columns_as_row:
        yield filter(row)


def windowed_view(arr, window, overlap):
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides).tolist()


def learn():
    # np.warnings.simplefilter(action='ignore', category=UserWarning)
    overlaped = 5
    # windows_size = 10
    # clusters = 5
    data_set = process_from_files()
    classes_names_as_is_in_data = create_classes_names_list(data_set)
    print(len(classes_names_as_is_in_data))
    files_as_nested_list = get_files_as_list_of_lists(data_set)
    print("start testing")
    for clusters in [5]:
        windows_sizes = [5]
        for windows_size in windows_sizes:
            # if windows_size == 5:
            #     overlaps = [1, 4]
            # elif windows_size == 10:
            #     overlaps = [1, 5, 9]
            # elif windows_size == 15:
            #     overlaps = [1, 7, 14]
            # elif windows_size == 20:
            #     overlaps = [1, 10, 19]
            # elif windows_size == 25:
            #     overlaps = [1, 13, 24]
            # elif windows_size == 30:
            #     overlaps = [1, 15, 29]
            # elif windows_size == 35:
            #     overlaps = [1, 19, 34]
            # elif windows_size == 40:
            #     overlaps = [1, 20, 39]
            for overlaped in [2]:
                X_train, X_test, _, y_test = train_test_split(files_as_nested_list, classes_names_as_is_in_data,
                                                              test_size=0.9, random_state=4564567, shuffle=True)

                # files_as_windows_train = get_files_as_windows(X_train, windows_size, overlaped)
                files_as_windows_test = get_files_as_windows(X_test, windows_size, overlaped)

                all_sliding_windows = get_all_sliding_windows(X_train, windows_size, overlaped)
                kmeans_models = prepare_codebook(all_sliding_windows, clusters)

                histograms_test = get_histogram_basic_on_kmean(clusters, kmeans_models, files_as_windows_test)

                X_train, X_test, y_train1, y_test1 = train_test_split(histograms_test, y_test,
                                                                      test_size=0.3, random_state=456547, shuffle=True)

                # find_the_best(X_train, X_test, y_train1, y_test1)

                # # # different svm
                models = []
                models.append(('kneighboard ', KNeighborsClassifier(
                    algorithm='auto',
                    leaf_size=30,
                    metric='manhattan',
                    metric_params=None,
                    n_jobs=1,
                    n_neighbors=1,
                    p=1,
                    weights='distance')))
                models.append(('extra tree entropy', ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                                       max_depth=None, max_features='log2',
                                                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                       min_impurity_split=None, min_samples_leaf=1,
                                                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                       n_estimators=1024, n_jobs=1, oob_score=False, random_state=1,
                                                       verbose=False, warm_start=False)))
                models.append(('random trees',
                               sklearn.ensemble.RandomForestClassifier(bootstrap=True, class_weight=None,
                                                                       criterion='entropy',
                                                                       max_depth=None, max_features='log2',
                                                                       max_leaf_nodes=None,
                                                                       min_impurity_decrease=0.0,
                                                                       min_impurity_split=None,
                                                                       min_samples_leaf=1, min_samples_split=2,
                                                                       min_weight_fraction_leaf=0.0, n_estimators=231,
                                                                       n_jobs=1,
                                                                       oob_score=False, random_state=2, verbose=False,
                                                                       warm_start=False)))
                #
                models.append(('svm poly', svm.SVC(kernel='rbf', gamma=1.0, C=0.001)))
                models.append(('gaussian nb', GaussianNB()))

                # results = []
                # names = []
                # # seed = 7
                # # scoring = 'accuracy'
                # # from sklearn.feature_selection import RFE
                rezultaty = ""
                for name, model in models:
                    # # print('start testing model {}'.format(name))
                    # # rfe = PCA(model)
                    # # rfe = rfe.fit(X_train, y_train)
                    # # # summarize the selection of the attributes
                    # # print(rfe)
                    # # print(rfe.ranking_)
                    kfold = model_selection.RepeatedKFold(n_splits=5, random_state=7456456, n_repeats=10)
                    # print(name)
                    selection = svc_param_selection(histograms_test, y_test, kfold, model, name)
                    print(selection)
                    # # model.fit(histograms_train, y_train)
                    # cv_results = model_selection.cross_val_score(model, histograms_test, y_test, cv=kfold, scoring='accuracy')
                    # # msg = " %f (%f)" % (cv_results.mean(), cv_results.std())
                    # # print(name+" ", msg)
                    # # rezultaty+=str(cv_results.mean())+","+str(cv_results.std())+","
                    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    # print(msg)
                    # # labels = ['ME', 'OFTEN', 'RARE','PAIN','HEAD',
                    # #           'TOOTH','THROAT','BACK','LUNGS','VOMIT',
                    # #           'COUGH','RUNNY NOSE','FEVER', 'COLD', 'SWEAT',
                    # #           'INFLAMMATION', 'VERY', 'LITTLE']
                    # #
                    # # model.fit(X_train, y_train1)
                    # # # print(histograms_test)
                    # label = model.predict(X_test)
                    # # from sklearn.metrics import accuracy_score
                    # # print('not know data', accuracy_score(y_test1, label))
                    # #
                    # # # pickle.dump(model, open('codebook_models/kmean_provisor.sav', 'wb'))
                    # # label = model.predict(X_test)
                    # #
                    # # # conf_mat = confusion_matrix(label, y_test1)
                    # # # plot_confusion_matrix(conf_mat, labels)
                    # print(clusters, ',', windows_size, ', ', overlaped, ',', rezultaty)
                    # print(msg)


def svc_param_selection(X, y, nfolds, kn, kernel_name):
    from sklearn.model_selection import GridSearchCV
    # scaler = StandardScaler()
    # X = preprocessing.scale(X)
    # X = scaler.fit_transform(X)
    C_range = 10. ** np.arange(-3, 4)
    gamma_range = 10. ** np.arange(-3, 4)
    # weight_options = ["uniform", "distance"]
    k_range = [1, 2, 4, 8, 16, 32, 64, 128]
    metrics = ['manhattan', 'minkowski', 'euclidean', 'chebyshev']

    max_features = ['auto', 'sqrt', 'log2']
    # classifier__criterion = ["gini", "entropy"]
    n_estimators = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # param_grid = {"max_features":max_features, 'n_estimators':n_estimators}
    # param_grid = {"n_neighbors":k_range, 'metric':metrics}
    param_grid = {"C": C_range, 'gamma': gamma_range}

    print('griduje')
    grid_search = GridSearchCV(kn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    score_dict = grid_search.grid_scores_

    # We extract just the scores
    scores = [x[1] for x in score_dict]

    scores = np.array(scores).reshape(len(gamma_range), len(C_range))
    # print(scores)
    # Make a nice figure
    # plt.figure(figsize=(8, 6))
    title = "Heatmap for {}".format('SVM kernel rbf Classifier')
    plt.title(title)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
    plt.ylabel('max features')
    plt.xlabel('number of classifier')
    plt.colorbar(ticks=[0, 0.3, 0.5, 0.7, 0.9, 1], label="precision")
    plt.clim(0, 1)
    plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
    plt.yticks(np.arange(len(gamma_range)), gamma_range)
    plt.savefig(kernel_name)
    print(grid_search.cv_results_)
    print(grid_search.best_score_)
    "poly 0.91734375 'gamma': 1.0, 'C': 0.001"
    "linear 0.79296875 , {'gamma': 0.001, 'C': 0.10000000000000001}"
    "rbf {'gamma': 0.01, 'C': 100.0} 0.92476563"
    " extra tree 0.935390625 {'n_estimators': 1024, 'max_features': 'log2'}" \
    "0.927734375 + {'criterion': 'entropy', 'n_estimators': 2048}"
    """0.92390625 {'n_neighbors': 2, 'metric': 'manhattan'} kneighboard"""
    return grid_search.best_params_


def get_files_as_windows(files_as_nested_list, windows_size, overlaped):
    files_as_windows = []
    for file in files_as_nested_list:

        file_windows = []
        for index, sensor in enumerate(file):
            windows = windowed_view(sensor, windows_size, overlaped)
            file_windows.append(windows)
        files_as_windows.append(file_windows)
    return files_as_windows


def get_all_sliding_windows(files_as_nested_list, windows_size, overlaped):
    all_sliding_windows = [[] for _ in range(len(files_as_nested_list[0]))]
    for file in files_as_nested_list:

        for index, sensor in enumerate(file):
            # sensor = sensor.values
            windows = windowed_view(sensor, windows_size, overlaped)
            all_sliding_windows[index].extend(windows)
    return all_sliding_windows


def get_histogram_basic_on_kmean(clusters_number, kmeans_models, files_as_windows):
    # print('get histogram')
    histograms = []
    for file in files_as_windows:
        file_histogram = []
        for index, sensor_data_as_windows in enumerate(file):
            predicted_list = kmeans_models[index].predict(sensor_data_as_windows)
            from collections import Counter
            histogram = dict(Counter(predicted_list))
            for key in range(0, clusters_number):
                if key not in histogram.keys():
                    histogram[key] = 0
            d = dict(sorted(histogram.items()))
            histogram = list(d.values())
            # normalized_histogram = [float(i) / sum(histogram) for i in histogram]
            file_histogram.extend(histogram)
        histograms.append(file_histogram)
    return histograms


def prepare_codebook(all_sliding_windows, number_of_clusters):
    # print('start prepare codebook')
    kmeans_models = []
    for index, sensor_all_windows in enumerate(all_sliding_windows):
        kmean_model = get_kmeans_model(sensor_all_windows, number_of_clusters)
        filename = '{}.sav'.format(SENSORS_NAMES[index])
        # pickle.dump(kmean_model, open('codebook_models/'+filename, 'wb'))
        kmeans_models.append(kmean_model)
    # print('end prepare codebook')
    return kmeans_models


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def find_the_best_autosklearn(X_train, X_test, y_train, y_test):
    import autosklearn.classification
    automl = autosklearn.classification.AutoSklearnClassifier(
        per_run_time_limit=100,
        tmp_folder='./models_data/autosklearn_cv_example_tmp',
        output_folder='./models_data/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(X_train.copy(), y_train.copy(), dataset_name='digits')
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(X_train.copy(), y_train.copy())

    # print(automl.show_models())
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    predictions = automl.predict(X_test)
    print(automl.show_models())
    print(automl.sprint_statistics())
    print(automl.cv_results_)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


def mlbox_counter():
    from mlbox.preprocessing import Reader, Drift_thresholder
    from mlbox.optimisation import Optimiser
    from mlbox.prediction import Predictor
    target_name = '601'

    rd = Reader(sep=",")
    df = rd.train_test_split(['train_egg.csv', 'test_egg.csv'], target_name)
    # print(df)
    dft = Drift_thresholder()
    df = dft.fit_transform(df)  # removing non-stable features (like ID,...)

    opt = Optimiser(scoring="accuracy", n_folds=10)
    space = {

        'est__strategy': {"search": "choice",
                          "space": ["LightGBM"]},
        'est__n_estimators': {"search": "choice",
                              "space": [150]},
        'est__colsample_bytree': {"search": "uniform",
                                  "space": [0.8, 0.95]},
        'est__subsample': {"search": "uniform",
                           "space": [0.8, 0.95]},
        'est__max_depth': {"search": "choice",
                           "space": [5, 6, 7, 8, 9]},
        'est__learning_rate': {"search": "choice",
                               "space": [0.07]}

    }
    best = opt.optimise(space, df, 15)

    prd = Predictor()
    prd.fit_predict(best, df)


def find_the_best(X_train, X_test, y_train, y_test):
    from hyperopt import tpe
    import hpsklearn
    import hpsklearn.demo_support
    import time
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    from hpsklearn import HyperoptEstimator, random_forest, svc, knn

    estimator = hpsklearn.HyperoptEstimator(
        preprocessing=[],
        classifier=knn('myknn'),
        algo=tpe.suggest,
        # trial_timeout=500.0,  # seconds
        max_evals=100,
    )

    fit_iterator = estimator.fit_iter(X_train, y_train)
    fit_iterator.next()
    plot_helper = hpsklearn.demo_support.PlotHelper(estimator,
                                                    mintodate_ylim=(-.01, .05))
    while len(estimator.trials.trials) < estimator.max_evals:
        fit_iterator.send(1)  # -- try one more model
        plot_helper.post_iter()

    plot_helper.post_loop()
    plt.show()
    # -- Model selection was done on a subset of the training data.
    # -- Now that we've picked a model, train on all training data.
    estimator.retrain_best_model_on_full_data(X_train, y_train)

    print('Best preprocessing pipeline:')
    for pp in estimator._best_preprocs:
        print(pp)
    print()
    print('Best classifier:\n', estimator._best_learner)

    print(estimator.best_model())
    test_predictions = estimator.predict(X_test)
    acc_in_percent = 100 * np.mean(test_predictions == y_test)
    print()
    print('Prediction accuracy in generalization is ', acc_in_percent)


def genetic_algorithm(X_train, X_test, y_train, y_test):
    from tpot import TPOTClassifier

    tpot = TPOTClassifier(generations=100, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))


def get_time_window(nested_file, rows_in_window):
    for index in range(10, len(nested_file), rows_in_window):
        yield nested_file[index - 10:index + rows_in_window]


def get_kmeans_model(clusters, clusters_group):
    # min_kmean = None
    # min_sum_of_kmean = 100000000
    # for i in range(0, 1):
    return KMeans(n_clusters=clusters_group).fit(clusters)
    #     sum_euclidean = 0
    #     for cluster1, cluster2 in itertools.combinations(kmeans.cluster_centers_, 2):
    #         sum_euclidean += euclidean(cluster1, cluster2)
    #     if sum_euclidean < min_sum_of_kmean:
    #         min_kmean = kmeans.labels_
    # return min_kmean


def get_created_group_as_flat_list(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def file_as_one_row(nested_file):
    flat_rows = []
    for row in nested_file:
        flat_rows.extend(row)
    return flat_rows


def create_classes_names_list(training_set):
    """
    :param training_set: dict(list, list)
    :return: (list, list)
    """
    learn_classes_list = []
    for k, v in training_set.items():
        learn_classes_list.extend([str(k)] * len(v))
    return learn_classes_list


def get_flat_list_as_nested(file_as_one_row):
    return list(zip(*[iter(file_as_one_row)] * 18))


def get_files_as_list_of_lists(data_set):
    """
    :param data_set: dict(list[list], list[[list])
    :return: list(list)
    """
    files = []
    for k, v in data_set.items():
        files.extend(v)
    return files


def separate_data_set_to_testing_and_teaching(data_separated_by_class):
    """

    :param data_set:list[list[list]]
    :return: list[list]
    """
    learn_classes_list = []
    test_classes_list = []
    for class_files in data_separated_by_class:
        pass
        #     training_number_of_classes = int(len(files)*TEACHING_DATA_PERCENT)
        #     testing_number_of_classes = int(len(files)*TESTING_DATA_PERCENT)
        #     get_nested_list_as_flat_list
        #     learn_classes_list.extend(v[:training_number_of_classes])
        #     test_classes_list.extend(v[-testing_number_of_classes:])
        # return learn_classes_list, test_classes_list
        # """
        # :param data_set: dict(list, list)
        # :return: (list[list], list[list])
        # """
        # learn_classes_list = []
        # test_classes_list = []
        # for k, v in data_set.items():
        #     training_number_of_classes = int(len(v)*TEACHING_DATA_PERCENT)
        #     testing_number_of_classes = int(len(v)*TESTING_DATA_PERCENT)
        #
        #     learn_classes_list.extend(v[:training_number_of_classes])
        #     test_classes_list.extend(v[-testing_number_of_classes:])
        # return learn_classes_list, test_classes_list


if __name__ == "__main__":
    learn()
# PCA(copy=True, iterated_power='auto', n_components=8, random_state=None,
#     svd_solver='auto', tol=0.0, whiten=True)
# ()
# ('Best classifier:\n', ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
#                                             max_depth=None, max_features='sqrt', max_leaf_nodes=None,
#                                             min_impurity_decrease=0.0, min_impurity_split=None,
#                                             min_samples_leaf=1, min_samples_split=2,
#                                             min_weight_fraction_leaf=0.0, n_estimators=156, n_jobs=1,
#                                             oob_score=False, random_state=3, verbose=False,
#                                             warm_start=False))
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
#                                  max_depth=None, max_features='sqrt', max_leaf_nodes=None,
#                                  min_impurity_decrease=0.0, min_impurity_split=None,
#                                  min_samples_leaf=1, min_samples_split=2,
#                                  min_weight_fraction_leaf=0.0, n_estimators=156, n_jobs=1,
#                                  oob_score=False, random_state=3, verbose=False,
#                                  warm_start=False),
#  'preprocs': (PCA(copy=True, iterated_power='auto', n_components=8, random_state=None,
#                   svd_solver='auto', tol=0.0, whiten=True),), 'ex_preprocs': ()}
# ()

# Best preprocessing pipeline:
# MinMaxScaler(copy=True, feature_range=(0.0, 1.0))
# ()
# ('Best classifier:\n', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
#            metric_params=None, n_jobs=1, n_neighbors=2, p=1,
#            weights='distance'))
# {'learner': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
#            metric_params=None, n_jobs=1, n_neighbors=2, p=1,
#            weights='distance'), 'preprocs': (MinMaxScaler(copy=True, feature_range=(0.0, 1.0)),), 'ex_preprocs': ()}
# ()
# ('Prediction accuracy in generalization is ', '75.22935779816514')
# --------------------------------------------------
# ('Best classifier:\n', SVC(C=80734.0273761, cache_size=512, class_weight=None, coef0=0,
#   decision_function_shape='ovr', degree=4.0, gamma=818.46647261,
#   kernel='poly', max_iter=278593425.0, probability=False, random_state=4,
#   shrinking=False, tol=0.000639665051648, verbose=False))
# {'learner': SVC(C=80734.0273761, cache_size=512, class_weight=None, coef0=0,
#   decision_function_shape='ovr', degree=4.0, gamma=818.46647261,
#   kernel='poly', max_iter=278593425.0, probability=False, random_state=4,
#   shrinking=False, tol=0.000639665051648, verbose=False), 'preprocs': (Normalizer(copy=True, norm='l2'),), 'ex_preprocs': ()}
# ()
# ('Prediction accuracy in generalization is ', '76.87074829931973')
# ('Best classifier:\n', ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
#            max_depth=None, max_features=0.171769356288,
#            max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=1650, n_jobs=1, oob_score=False, random_state=3,
#            verbose=False, warm_start=False))
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
#            max_depth=None, max_features=0.171769356288,
#            max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=1650, n_jobs=1, oob_score=False, random_state=3,
#            verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
# ()
# ('Prediction accuracy in generalization is ', '84.34782608695653')
# ('Best classifier:\n', ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=None, max_features=0.441871701219,
#            max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=2,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=1442, n_jobs=1, oob_score=False, random_state=2,
#            verbose=False, warm_start=False))
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=None, max_features=0.441871701219,
#            max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=2,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=1442, n_jobs=1, oob_score=False, random_state=2,
#            verbose=False, warm_start=False), 'preprocs': (PCA(copy=True, iterated_power='auto', n_components=32, random_state=None,
#   svd_solver='auto', tol=0.0, whiten=True),), 'ex_preprocs': ()}
# ()
# ('Prediction accuracy in generalization is ', '6.9364161849710975')
