import main
from sklearn import svm
from sklearn import neighbors
from collections import OrderedDict
import nltk

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    for lexelt in data:
        context_words = []
        for instance in data[lexelt]:
            context_words += nltk.word_tokenize(instance[1])[-window_size:]
            context_words += nltk.word_tokenize(instance[3])[:window_size]
        s[lexelt] = list(set(context_words))
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    for instance in data:
        context = []
        labels[instance[0]] = instance[4]
        context += nltk.word_tokenize(instance[1])[-window_size:]
        context += nltk.word_tokenize(instance[3])[:window_size]
        word_counts = []
        for word in s:
            word_counts.append(context.count(word))
        vectors[instance[0]] = word_counts
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    # --------- SVM ----------------
    svm_clf = svm.LinearSVC()
    X = []
    y = []
    for inst in X_train:
        X.append(X_train[inst])
        y.append(y_train[inst])

    svm_clf.fit(X, y)

    X2 = []
    X_test2 = OrderedDict(X_test)
    for inst in X_test2:
        X2.append(X_test2[inst])

    results_svm = svm_clf.predict(X2)
    keys = list(X_test2.keys())
    svm_results = zip(keys, results_svm)
    # --------------------------------------

    # ----------- KNN ----------------------
    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(X, y)
    results_knn = knn_clf.predict(X2)
    knn_results = zip(keys, results_knn)


    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    with open(output_file, 'w+') as f:
        for lexelt in sorted(results):
            results_sorted = sorted(results[lexelt], key=lambda x: int(x[0].split(".")[-1]))
            for result in results_sorted:
                f.write(main.replace_accented(lexelt + " " + result[0] + " " + result[1] + "\n"))
        lines = f.readlines()
        lines.sort()
        for line in lines:
            f.write(line)

        # with open(output_file, 'w+') as f:
        #     for lexelt in sorted(results):
        #         results_sorted = sorted(results[lexelt], key=lambda x: x[0])
        #         for result in results_sorted:
        #             f.write(main.replace_accented(lexelt + " " + result[0] + " " + result[1] + "\n"))
        #     lines = f.readlines()
        #     lines.sort()
        #     for line in lines:
        #         f.write(line)
# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)


    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



