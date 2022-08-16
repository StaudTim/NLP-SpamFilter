import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class SpamFilter:
    def __init__(self):
        self.data = pd.DataFrame
        self.spam = []
        self.ham = []
        self.prob_word_in_spam = {}
        self.prob_word_in_ham = {}
        self.prob_label = {}
        self.smoothing = 1
        self.punctuation = ['!', '"', ',', ';', '?', ':', '(', ')', '[', ']', '+']
        self.stopwords = stopwords.words('german')
        self.stemmer = SnowballStemmer('german')

    def _preprocess_data(self, data):
        # ignore upper and lower case and delete punctuation
        data['text'] = data['text'].str.lower()
        data['label'] = data['label'].str.lower()
        for i in self.punctuation:
            data['text'] = data['text'].str.replace(i, '', regex=True)
        return data

    def _split_data(self):
        # split into spam/ham and delete stopwords
        for i, row in self.data.iterrows():
            for word in row['text'].split(' '):

                # append the stem of the word to the list
                if word not in self.stopwords:
                    stem = self.stemmer.stem(word)
                    if row['label'] == 'spam':
                        self.spam.append(stem)
                    else:
                        self.ham.append(stem)

    def _calc_prob_label(self):
        # calculate the probability for all labels e.g. P(SPAM)
        count_sentences = self.data.shape[0]
        count_labels = self.data.shape[1]
        denominator = count_sentences + self.smoothing * count_labels

        ham, spam = self.data['label'].value_counts()
        self.prob_label['ham'] = (ham + self.smoothing) / denominator
        self.prob_label['spam'] = (spam + self.smoothing) / denominator

    def _calc_prob_word_in_label(self, words):
        count_different_words = len(set(self.spam + self.ham))

        # calculate the probability that the word is spam/ham e.g. P("SECRET"|SPAM)
        for word in words:
            count = 0
            for spam in self.spam:
                if word == spam:
                    count += 1
            numerator = count + self.smoothing
            denominator = len(self.spam) + self.smoothing * count_different_words
            self.prob_word_in_spam[word] = numerator / denominator

            count = 0
            for ham in self.ham:
                if word == ham:
                    count += 1
            numerator = count + self.smoothing
            denominator = len(self.ham) + self.smoothing * count_different_words
            self.prob_word_in_ham[word] = numerator / denominator

    def _law_of_total_probability(self, words, numerator):
        denominator = self.prob_label['ham']
        for word in words:
            denominator *= self.prob_word_in_ham[word]
        return denominator + numerator

    def _bayes_rule(self, words):
        # calculate numerator -> P(Text|Label * P(Label)
        numerator = self.prob_label['spam']
        for word in words:
            numerator *= self.prob_word_in_spam[word]

        # calculate denominator -> P(Text|b) * P(b) where b = {spam, ham}
        denominator = self._law_of_total_probability(words, numerator)

        # calculate probability
        probability = numerator / denominator
        return probability

    def fit(self, data):
        self.data = self._preprocess_data(data)
        self._split_data()

    def predict(self, data):
        test_set = self._preprocess_data(data)
        df = pd.DataFrame({'label': [],
                           'text': [],
                           'predicted_label': []}
                          )

        # iterate through testdata, remove stopwords and apply stemming
        for i, row in test_set.iterrows():
            words = []
            for word in row['text'].split(' '):
                if word not in self.stopwords:
                    stem = self.stemmer.stem(word)
                    words.append(stem)

            # check if there is a word which doesn't appear in spam/ham -> laplacian smoothing necessary
            words_found = 0
            for word in words:
                for spam in self.spam:
                    if word == spam:
                        words_found += 1
                        break

                for ham in self.ham:
                    if word == ham:
                        words_found += 1
                        break

            self.smoothing = 1  # factor for laplacian smoothing
            if words_found == 2 * len(words):
                self.smoothing = 0

            # calculate the probability for the sentence to be spam
            self._calc_prob_label()
            self._calc_prob_word_in_label(words)
            prob_for_spam = self._bayes_rule(words)

            predicted_label = 'ham'
            if prob_for_spam >= 0.5:
                predicted_label = 'spam'
            df.loc[len(df)] = [row["label"], row["text"], predicted_label]
        return df
