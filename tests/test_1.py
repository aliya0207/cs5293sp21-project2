import pytest
import main
import nltk
import spacy

text = " Daniel Radcliffe was exceptional in the last movie of the series."

def test1():
    name = main.get_redacted_entity(text)
    assert name is not None    
