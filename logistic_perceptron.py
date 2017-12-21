import sys
import os
import string
from nltk.stem import PorterStemmer
import math

# fetch command line arguments
stopwords_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]

ps = PorterStemmer()

def getpaths():  # fetch all files from the directory
    train_ham_files =[]
    train_spam_files =[]
    test_ham_files =[]
    test_spam_files=[]
    stopwords_file = None

    for root,dirs,files in os.walk(train_dir):
        for dir in dirs:
            if "ham" in dir.lower():
                directory_ham = os.path.join(train_dir, dir)
                for file in os.listdir(directory_ham):
                    if file.endswith(".txt"):
                        train_ham_files.append((os.path.join(directory_ham, file)))
        for dir in dirs:
            if "spam" in dir.lower():
                directory_spam = os.path.join(test_dir, dir)
                for file in os.listdir(directory_spam):
                    if file.endswith(".txt"):
                        train_spam_files.append((os.path.join(directory_spam, file)))
    for root, dirs, files in os.walk(test_dir):
        for dir in dirs:
            if "ham" in dir.lower():
                directory_ham = os.path.join(test_dir, dir)
                for file in os.listdir(directory_ham):
                    if file.endswith(".txt"):
                        test_ham_files.append((os.path.join(directory_ham, file)))
        for dir in dirs:
            if "spam" in dir.lower():
                directory_spam = os.path.join(test_dir, dir)
                for file in os.listdir(directory_spam):
                    if file.endswith(".txt"):
                        test_spam_files.append((os.path.join(directory_spam, file)))

    for root,dirs,files in os.walk(stopwords_dir):
        stopword_file_name = "stopWords.txt"
        for file in files:
            if file == stopword_file_name:
                stopwords_file = os.path.join(root,file)
                stopwords_file.replace("/", "\\")


    return train_ham_files, train_spam_files, test_ham_files, test_spam_files, stopwords_file

def load_stopwords(stopword_path):
    with open(stopword_path, 'r', encoding='utf8', errors='ignore') as f:
        stopwords_list = f.read().split()
    return stopwords_list

def readFiles(files, total_words,stopword_list): #read content of all files and create data matrix
    data =[]
    for file in files:
        try:
            with open(file, encoding='utf8', errors='ignore') as f:
                file_content = [word.strip(string.punctuation) for line in f for word in line.lower().split()]
                file_filtered_content = list(filter(None, file_content))

                file_dict={}
                for word in file_filtered_content:

                    if word in stopword_list:  # remove stopwords from each of the directory
                        continue

                    if file_dict.get(word, 0) == 0:
                        file_dict[word] = 1
                    else:
                        file_dict[word] += 1

            row = {total_words.index(index) + 1: file_dict[index] for index in file_dict}  #convert to index of words based mapping
            row[0] = 1
            data.append(row)
        except IOError as e:
            print("ERROR:{}".format(e));
    return data

def accuracy(correct_prediction, actual_label):
    total = len(correct_prediction)
    right = sum(int(correct_prediction[i] == actual_label[i]) for i in range(total))
    return 100*float(right) / float(total)

def transpose(list_dict, no_of_coulms): #get transpose of matrix
    transposed = []
    for j in range(no_of_coulms):
        row = {i: row[j] for i, row in enumerate(list_dict) if j in row}
        transposed.append(row)
    return transposed

def logisticRegressionTrain(total_train, train_class_labels, words, total_test, test_class_labels):

    print("Logistic Regression based Classification")
    total_train_transpose = transpose(total_train, words) #get transpose for summation of feature and weight vectors
    weights = [0] * words # initialise weights to 0

    for regularizor in [0.001, .01, .1, 1, 100]:
        no_of_iter = 100
        learning_rate = 0.005

        for k in range(no_of_iter):
            error = calculateError(total_train, weights, train_class_labels)
            gradient = calculateGradient(learning_rate,total_train_transpose,error,words)
            l2Value = calculateL2(learning_rate,regularizor,weights,words)
            weights = updateWeights(weights,gradient,l2Value,words)

        predicted_labels = logisticRegressionTest(total_test,weights)
        print("RegularizationConstant = {}  accuracy = {:.6f}".format(regularizor, accuracy(predicted_labels, test_class_labels)))

def getAllWords(allDirs): # get all unique words in train/test spam/ham directory
    total_2D_Words = []
    total_words = []

    for dirs in allDirs:
        for file in dirs:
            try:
                with open(file, encoding='utf8', errors='ignore') as f:
                    file_content = [word.strip(string.punctuation) for line in f for word in line.lower().split()]
                    file_filtered_content = list(filter(None, file_content))
                    total_2D_Words.append(file_filtered_content)
            except IOError as e:
                print("ERROR:{}".format(e));

    for files in total_2D_Words:
        for content in files:
            total_words.append(content)
    return list(set(total_words))

def sigmoid(x):  # throws overflow error if not handled
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0

def calculateError(total_train, weights,class_labels): #calculate error from actual labels
    return [sigmoid(feature_weight_product(total_train[i], weights)) - class_labels[i] for i in range(len(total_train))]

def calculateGradient(learning_rate,total_train_transpose,error,words): #calculate the differential amount
    return [learning_rate * feature_weight_product(total_train_transpose[j], error) for j in range(words)]

def calculateL2(learning_rate,regularizor,weights,words): # calculate lambda
    return [learning_rate * regularizor * weights[j] for j in range(words)]

def updateWeights(weights,gradient,l2Value,words):
    return [weights[j] - gradient[j] + l2Value[j] for j in range(words)]

def feature_weight_product(features, weights):
    return sum(weights[i] * features[i] for i in features)

def logisticRegressionTest(total_test,weights):
    prob = [sigmoid(feature_weight_product(total_test[i], weights)) for i in range(len(total_test))]
    return [int(probability > 0.5) for probability in prob]

def perceptronTest(total_test, weights):
    return [int(feature_weight_product(total_test[i], weights) > 0) for i in range(len(total_test))]

def perceptronTrain(total_train, train_class_labels, words, total_test, test_class_labels):
    print("Perceptron based Classification")

    for no_of_iterations in [5,10,20,50,100]:
        for learning_rate in [0.001, 0.01, 0.1, 1, 10]:
            weights = [0] * words

            for k in range(no_of_iterations):
                for i in range(len(total_train)):

                    perceptron_output = int(feature_weight_product(total_train[i], weights) > 0)
                    for j in total_train[i]:
                        weights[j] += learning_rate * (train_class_labels[i] - perceptron_output) * total_train[i][j]

            predicted_labels = perceptronTest(total_test, weights)
            print("no of iterations = {}   learning rate = {}   accuracy = {:.5f}".format(no_of_iterations, learning_rate, accuracy(predicted_labels, test_class_labels)))


def main():

    # get all files from directory
    train_ham_files = []
    train_spam_files = []
    test_ham_files = []
    test_spam_files = []
    train_ham_files, train_spam_files, test_ham_files, test_spam_files, stopwords_list = getpaths()
    stopword_list = load_stopwords(stopwords_list) # get stopWordList

    total_words_in_train_list = getAllWords([train_ham_files, train_spam_files, test_ham_files, test_spam_files])

    # read files and create data matrix for each dataset
    train_ham = readFiles(train_ham_files,total_words_in_train_list,stopword_list)
    train_spam = readFiles(train_spam_files,total_words_in_train_list,stopword_list)
    test_ham = readFiles(test_ham_files,total_words_in_train_list,stopword_list)
    test_spam = readFiles(test_spam_files,total_words_in_train_list,stopword_list)

    total_train = train_ham + train_spam
    train_class_labels = [0] * len(train_ham) + [1] * len(train_spam)

    total_test = test_ham + test_spam
    test_class_label = [0] * len(test_ham) + [1] * len(test_spam)

    # LR
    logisticRegressionTrain(total_train, train_class_labels, len(total_words_in_train_list)+ 1,total_test, test_class_label)

    #Perceptron
    perceptronTrain(total_train, train_class_labels, len(total_words_in_train_list)+ 1,total_test, test_class_label)

if __name__ == '__main__':
    main()