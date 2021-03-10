# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Choose candidates for terminolgy extraction
and do some preprocessing.
"""
from nltk import bigrams
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist


class Preprocess:

    """
    A class that does some processing of a corpus. A corpus can be
    an nltk corpus or directory of text files.

    Attributes:
        corpus

    Methods:
        is_lexical(word_i, word_j):
            Check if both words are alpha-numerical.
        candidates(min_prob=-15, stops=None):
            Get set of possible bigrams for terminology extraction.
        get_frequency(bigram_list, fileid=None):
            Get frequency of bigrams in bigram list in corpus or file.
        bigrams(fileid=None):
            Bigrams with frequency in whole corpus or file.
    """

    def __init__(self, corpus):
        """
        Constructs a preprocess instance.

        Args:
            corpus:
                Should either be the name of a directory with text files
                or a nltk corpus.

        Returns:
            None.
        """
        if isinstance(corpus, str):
            # Convert directory to Plaintext Corpus.
            corpus = PlaintextCorpusReader(corpus, ".*")
        self.corpus = corpus
        self._bigrams = FreqDist()
        self._count()

    @property
    def sum_bigrams(self):
        """Sum of frequency of all bigrams in corpus."""
        freq_bigram = self.bigrams()
        return sum(freq_bigram[bigram] for bigram in freq_bigram)

    def _count(self):
        """Counts occurences of bigrams in corpus, case insensitive.

        Returns:
            None.
        """
        print("Processing corpus...")
        words = [word.lower() for word in self.corpus.words()]
        bigrams_words = bigrams(words)
        for bigram in bigrams_words:
            self._bigrams[bigram] += 1

    @staticmethod
    def is_lexical(word_i, word_j):
        """Checks if two words only contain alpha-numeric characters.

        Args:
            word_i (str): A string to be checked.
            word_j (str): A string to be checked.

        Returns:
            bool:
                True if both token are alpha-numeric,
                False otherwise.
        """
        if word_i.isalnum() and word_j.isalnum():
            return True
        return False

    def candidates(self, min_count=4, stops=None):
        """
        Generate a list of possible candidates for terminology extraction.

        A bigram is considered a candidate if it exceeds a minimum proportion
        of whole corpus, only contains alpha-numeric characters and doesn't
        contain tokens in stopword list.

        Args:
            min_count (int):
                Minimum frequency a bigram has to have to be considered a
                candidate. Absolute frequency are used.
            stops (list):
                List of strings. If a bigram contains a word of that list, it
                is not considered a candidate. If default is used,
                an empty list is used. Default is None.

        Returns:
            set:
                set of tuples containing two strings.
        """
        if stops is None:
            stops = []
        candidates = set()
        for word_i, word_j in self.bigrams():
            # Filter out bigrams with stopwords.
            if word_i not in stops and word_j not in stops:
                # Make sure bigrams only alpha-numeric.
                if self.is_lexical(word_i, word_j):
                    # Filter out infrequent bigrams.
                    if self.bigrams()[word_i, word_j] >= min_count:
                        candidates.add((word_i, word_j))
        return candidates

    def get_frequency(self, bigram_list, fileid=None):
        """Get the frequency of a list of bigrams

        Either get frequency for the whole corpus or for
        a specific file. Bigrams that don't occur in corpus/file
        are not keys in returned dictionaries.

        Args:
            bigram_list (list):
                List with two-tuples of strings
            fileid (str):
                Id of file in corpus. If default is used, gets frequency
                in whole corpus. Default is None.

        Returns:
            dict:
                Keys are tuples of strings, values are frequencies in
                file/corpus (int)
        """
        freq = self.bigrams(fileid)
        return {bigr: freq[bigr] for bigr in bigram_list if bigr in freq}

    def bigrams(self, fileid=None):
        """Frequency of bigrams in file or corpus.

        Args:
            fileid (str):
                Id of file in corpus. If default is used, returns bigrams
                in whole corpus. Default is None.

        Returns:
            FreqDist:
                two-tuples of strings are keys, frequency in corpus/file are
                values.

        Raises:
            AssertionError:
                If given file is not in corpus.
        """
        if fileid is not None:
            # Make sure file is in corpus.
            assert fileid in self.corpus.fileids(), "File not in corpus."
            # Case insensitive.
            file_words = [word.lower() for word in self.corpus.words(fileid)]
            bigrams_file = bigrams(file_words)
            return FreqDist(bigrams_file)
        return self._bigrams


if __name__ == "__main__":
    pass
