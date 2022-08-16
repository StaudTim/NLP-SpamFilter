import pytest


def test_preprocess_data(fxt_get_df, fxt_naive_bayes, fxt_get_df_preprocessed):
    # Arrange
    naive_bayes = fxt_naive_bayes
    result = fxt_get_df_preprocessed

    # Act
    naive_bayes.data = naive_bayes._preprocess_data(fxt_get_df)

    # Assert
    assert naive_bayes.data.equals(result)


def test_split_data(fxt_get_df_preprocessed, fxt_naive_bayes):
    # Arrange
    naive_bayes = fxt_naive_bayes
    naive_bayes.data = fxt_get_df_preprocessed
    result_stem_spam = ['angebot', 'geheim', 'klick', 'geheim', 'link', 'geheim', 'sport', 'link']
    result_stem_ham = ['spiel', 'sport', 'heut', 'geh', 'spiel', 'sport', 'geheim', 'sport', 'veranstalt', 'sport',
                       'heut', 'sport', 'kostet', 'geld']
    # Act
    naive_bayes._split_data()

    # Assert
    assert naive_bayes.spam == result_stem_spam and naive_bayes.ham == result_stem_ham


@pytest.mark.parametrize('smoothing, expected', [(0, {'ham': 0.625, 'spam': 0.375}), (1, {'ham': 0.6, 'spam': 0.4})])
def test_calc_prob_label(fxt_get_df_preprocessed, fxt_naive_bayes, smoothing, expected):
    # Arrange
    naive_bayes = fxt_naive_bayes
    naive_bayes.data = fxt_get_df_preprocessed
    naive_bayes.smoothing = smoothing
    result = expected

    # Act
    naive_bayes._calc_prob_label()

    # Assert
    assert naive_bayes.prob_label == result


@pytest.mark.parametrize('smoothing, word, spam, ham',
                         [(0, 'geheim', {'geheim': 0.375}, {'geheim': 1 / 14}),
                          (1, 'heute', {'heute': 1 / 19}, {'heute': 3 / 25})])
def test_calc_prob_word_in_label(fxt_get_classes, fxt_naive_bayes, smoothing, word, spam, ham):
    # Arrange
    naive_bayes = fxt_naive_bayes
    naive_bayes.spam, naive_bayes.ham = fxt_get_classes
    naive_bayes.smoothing = smoothing
    words = [word]

    # Act
    naive_bayes._calc_prob_word_in_label(words)

    # Assert
    assert naive_bayes.prob_word_in_spam == spam and naive_bayes.prob_word_in_ham == ham
