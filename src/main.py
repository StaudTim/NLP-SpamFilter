from spam_filter import SpamFilter
import pandas as pd

'''
Let's make the following definitions:
    - 'spam' is a positive class
    - 'ham' is a negative class
'''


def calc_accuracy(labeled_data):
    """
    Defined as the number of true positives plus true negatives over the number of true positives plus true negatives
    """
    correct_rows = 0
    for i, row in labeled_data.iterrows():
        if row['label'] == row['predicted_label']:
            correct_rows += 1
    return round(correct_rows / len(labeled_data), 3)


def calc_precision(labeled_data):
    """
    Defined as the number of true positives over the number of true positives plus the number of false positives
    """
    true_positive = 0
    false_positive = 0
    for i, row in labeled_data.iterrows():
        if row['predicted_label'] == 'spam':
            if row['label'] == 'spam':
                true_positive += 1
            else:
                false_positive += 1
    return round(true_positive / (true_positive + false_positive), 3)


def calc_recall(labeled_data):
    """
    Defined as the number of true positives over the number of true positives plus the number of false negatives
    """
    true_positive = 0
    false_negative = 0
    for i, row in labeled_data.iterrows():
        if row['label'] == 'spam':
            if row['predicted_label'] == 'spam':
                true_positive += 1
            else:
                false_negative += 1
    return round(true_positive / (true_positive + false_negative), 3)


def calculate_f1(precision, recall):
    """
    Harmonic mean of precision and recall
    """
    return round(2 * precision * recall / (precision + recall), 3)


if __name__ == "__main__":
    data = pd.read_csv('dataset.csv', sep=';')

    # split into train and test set
    data_randomized = data.sample(frac=1, random_state=1)
    index = round(len(data_randomized) * 0.8)
    train_set = pd.DataFrame(data_randomized[:index])
    test_set = pd.DataFrame(data_randomized[index:])

    # train spam filter
    sf = SpamFilter()
    sf.fit(train_set)

    # test spam filter with test set and evaluate classifier output quality
    results = sf.predict(test_set)
    accuracy = calc_accuracy(results)
    precision = calc_precision(results)
    recall = calc_recall(results)
    f1 = calculate_f1(precision, recall)

    print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {precision}')
