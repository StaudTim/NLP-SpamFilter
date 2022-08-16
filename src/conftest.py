import pytest
import pandas as pd
from spam_filter import SpamFilter


@pytest.fixture()
def fxt_get_df():
    data = {'label': ['spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham'],
            'text': ['AngeBot ist, geheim!', 'Klicke geheim Link', 'Geheim, sport link', 'Spiel SPORT heute?',
                     'geh spiel sport', 'GEHEIM sport Veranstaltung', 'sport ist heute', 'sport kostet Geld!']}
    data_frame = pd.DataFrame(data)
    return data_frame


@pytest.fixture()
def fxt_naive_bayes():
    return SpamFilter()


@pytest.fixture()
def fxt_get_df_preprocessed():
    data = {'label': ['spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham'],
            'text': ['angebot ist geheim', 'klicke geheim link', 'geheim sport link',
                     'spiel sport heute',
                     'geh spiel sport', 'geheim sport veranstaltung', 'sport ist heute',
                     'sport kostet geld']}
    data_frame = pd.DataFrame(data)
    return data_frame


@pytest.fixture()
def fxt_get_classes():
    spam = ['angebot', 'geheim', 'klicke', 'geheim', 'link', 'geheim', 'sport', 'link']
    ham = ['spiel', 'sport', 'heute', 'geh', 'spiel', 'sport', 'geheim', 'sport', 'veranstaltung', 'sport',
           'heute', 'sport', 'kostet', 'geld']
    return spam, ham
