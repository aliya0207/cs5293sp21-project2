import pytest
import main
import nltk
import spacy

text = " Daniel Radcliffe was exceptional in the last movie of the series. "
def test2():
    name = main.get_redacted_entity(text)
    
    # there is 1 name in this file [ Daniel Radcliffe]
    assert len(name) == 1
