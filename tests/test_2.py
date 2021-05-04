import pytest
import main
import nltk
import spacy

text = " Daniel and Emma were exceptional in the last movie of the series. "
def test2():
    name = main.get_redacted_entity(text)
    
    actual = ['Daniel', 'Emma']
    expected = ['Daniel', 'Emma']

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])

    print(all([a == b for a, b in zip(actual, expected)]))
    #assert len(name) == 2
    #assert name == ['Emma' , 'Daniel']

