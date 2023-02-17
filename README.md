# Phrase book

##Extracting keywords from annotated sentences
First experiments to extract keywords, including some statistics, TF-IDF, rake, yake.  <br>
Preprocessing pipeline can be found in preprocessing.py. Extraction of main clauses not implemented yet.

### How to use the script
Adjust what you'd like to print out, then run: 
```
python3 extract_keywords.py
```
###Prepare environment
Install spacy small model for preprocessing:
```
pip install -U pip setuptools wheel 
pip install -U spacy 
python3 -m spacy download de_core_news_sm
```
Packages to install: 

```
pip install trafilatura
pip install summa
pip install git+https://github.com/smirnov-am/pytopicrank.git#egg=pytopicrank
pip install git+https://github.com/LIAAD/yake
pip install keyBERT

pip3 install rake-nltk
pip3 install summa
pip3 install git+https://github.com/LIAAD/yake
```
Install nltk 

```
pip install -U nltk 
```
If theres a problem with nltk stopwords, do this: 


```
import nltk
nltk.download('all')
or: 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```