import A
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn import neighbors
from sklearn import svm
from collections import OrderedDict
import string
from nltk.stem.lancaster import LancasterStemmer

# You might change the window size
window_size = 15

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}
    # print data
    # implement your code here

    # exclude = set(string.punctuation)
    st = LancasterStemmer()

    for instance in data:

        # left = nltk.word_tokenize(instance[1])[-2:]
        left = [st.stem(w) for w in nltk.word_tokenize(instance[1])[-2:]]
        # right = nltk.word_tokenize(instance[3])[:2]
        right = [st.stem(w) for w in nltk.word_tokenize(instance[3])[:2]]
        left_tagged = nltk.pos_tag(left)
        right_tagged = nltk.pos_tag(right)
        self_tag = nltk.pos_tag([instance[2]])

        labels[instance[0]] = instance[4]

        if len(left_tagged) > 1 and len(right_tagged) > 1:
            features[instance[0]] = {}

            # words themselves
            features[instance[0]]["W-2"] = left_tagged[0][0]
            features[instance[0]]["W-1"] = left_tagged[1][0]
            features[instance[0]]["W"] = self_tag[0][0]
            features[instance[0]]["W+1"] = right_tagged[0][0]
            features[instance[0]]["W+2"] = right_tagged[1][0]

            # pos-tags
            # features[instance[0]]["T-2"] = left_tagged[0][1]
            # features[instance[0]]["T-1"] = left_tagged[1][1]
            # features[instance[0]]["T"] = self_tag[0][1]
            # features[instance[0]]["T+1"] = right_tagged[0][1]
            # features[instance[0]]["T+2"] = right_tagged[1][1]


    # print features
    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []
    X = []
    y = []
    for inst in X_train:
        X.append(X_train[inst])
        y.append(y_train[inst])

    X2 = []
    X_test2 = OrderedDict(X_test)
    for inst in X_test2:
        X2.append(X_test2[inst])
    keys = list(X_test2.keys())
    # knn_clf = neighbors.KNeighborsClassifier()
    knn_clf = svm.LinearSVC()
    knn_clf.fit(X, y)
    knn_results = knn_clf.predict(X2)
    results = zip(keys, knn_results)

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)