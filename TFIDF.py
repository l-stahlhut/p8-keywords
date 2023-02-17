import math

def create_tf_matrix(sentences: list) -> dict:
    """
    For TF-IDF algorithm. Here document refers to a sentence.
    TF(t) = (Number of times the term t appears in a document) / (Total number of terms in the document)
    Creates dictionary with key = preprocessed sent and value = dictionary with {word: tf score, ...}
    """
    print('Creating tf matrix.')
    tf_matrix = {}
    for sentence in sentences:
        word_list = sentence.split()
        tf_table = {}
        words_count = len(word_list)
        # Determining frequency of words in the sentence
        word_freq = {}
        for word in word_list:
            word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1
        # Calculating tf of the words in the sentence
        for word, count in word_freq.items():
            tf_table[word] = count / words_count
        tf_matrix[sentence] = tf_table
    return tf_matrix


def create_idf_matrix(sentences: list) -> dict:
    """
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    """
    print('Creating idf matrix.')

    idf_matrix = {}

    documents_count = len(sentences)
    sentence_word_table = {}

    # Getting words in the sentence
    for sentence in sentences:
        clean_words = sentence.split()
        sentence_word_table[sentence] = clean_words

    # Determining word count table with the count of sentences which contains the word.
    word_in_docs = {}
    for sent, words in sentence_word_table.items():
        for word in words:
            word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1

    # Determining idf of the words in the sentence.
    for sent, words in sentence_word_table.items():
        idf_table = {}
        for word in words:
            idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix) -> dict:
    """
    Create a tf-idf matrix which is multiplication of tf * idf individual words
    """
    print('Calculating tf-idf of sentences.')

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix