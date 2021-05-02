# cs5293sp21-project2: THE UNREDACTOR

### Name: ALIYA SHAIKH
### EMAIL: aliyashaikh02@ou.edu

Whenever sensitive information is shared with the public, the data must go through a redaction process. That is, all sensitive names, places, and other sensitive information must be hidden. Documents such as police reports, court transcripts, and hospital records all containing sensitive information. Redacting this information is often expensive and time consuming.
In this project we will be creating an Unredactor. The unredactor will take redacted documents (that you generate) and the redaction flag as input, in return it will give the most likely candidates to fill in the redacted location. The unredactor only needs to unredact people names.
To discover the best names, we can have to train a model to help us predict missing words. For this assignment, you are expected to use the Large Movie Review Data Set. 
The data set is downloaded from : https://ai.stanford.edu/~amaas/data/sentiment/

## DESCRIPTION:

#### Redaction of the files:
1. Data Retrieval :Choose all text files that need to be redacted then each file is accessed as per index and the data is read by opening file in read mode.
2. Then used the nltk.word_tokenize to tokenize to words and Sentence Tokenizer to tokenize to sentences.
3. Then all the files are redacted (i.e Names of a person) are redacted from the files.
4. The redacted output files are then stored in a location which is mentioned in main as .redacted extension but having the same name.

#### Features of the names:
The extracted features are used for predicting the redacted name from the files. The features taken are:
1. length of the name: it is the length of the  redacted word. 
2. Number of redacted names : Number of names in the review.
3. Count of sentences = Number of sentences in given review. 
4. Count of words = Number of words in given review. 
5. count of characters = Number of characters in a given review. 
6. Number of spaces 

#### Model the classifier:
Used Dictvectorizer to get the feature. This is then used as an input to the model. Fit_transform is done on the training features. Output to the model are the redacted names. The model is fit by using Support Vector Classifier with probability. For every redacted word in the file, this will give a probability for each name in the training data. Then by using this model and probabilities, the top predicted names for the redacted names are displayed.

#### Tests:
test1.py:
In this test case we are testing if the function get_redacted_entity() is taking the data and extracting the names from it. After calling this function we see whether the number of names returned are greater than zero or not.


### REFERENCES:
1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
2. https://youtu.be/y_X4hXjTFNQ
3. http://www.itsyourip.com/scripting/python/python-remove-last-n-characters-of-a-string/
4. https://stackoverflow.com/questions/20290870/improving-the-extraction-of-human-names-with-nltk
