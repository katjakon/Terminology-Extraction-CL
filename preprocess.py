# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Choose candidates for terminolgy extraction
and do some preprocessing.
"""
import os

from nltk import bigrams
from nltk import pos_tag
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist


class Preprocess:

    DEMO = {"corpus": "demo/domain/"}

    """
    A class that does some processing of a corpus. A corpus can be
    an nltk corpus or directory of text files.

    Attributes:
        corpus: A nltk corpus object.

    Methods:
        corpus_stats:
            Prints some infos about given corpus.
        is_lexical(word_i, word_j):
            Check if both words are alphabetical.
        candidates(self, min_count=4, stops=None, tags={"NN", "NNP", "NNS"}):
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
            corpus = PlaintextCorpusReader(corpus, r".*\.txt")
        self.corpus = corpus
        self._bigrams = FreqDist()
        self._count()

    def _count(self):
        """Counts occurences of bigrams in corpus, case insensitive.

        Returns:
            None.
        """
        words = [word.lower() for word in self.corpus.words()]
        bigrams_words = bigrams(words)
        for bigram in bigrams_words:
            self._bigrams[bigram] += 1

    def corpus_stats(self):
        """Prints no of sentences, types and token in the corpus."""
        print("Number of sentences: {}".format(len(self.corpus.sents())))
        print("Token: {}".format(len(self.corpus.words())))
        types = FreqDist(self.corpus.words())
        print("Types: {}".format(len(types)))

    @staticmethod
    def is_lexical(word_i, word_j):
        """Checks if two words only contain alpha-numeric characters.

        Args:
            word_i (str): A string to be checked.
            word_j (str): A string to be checked.

        Returns:
            bool:
                True if both token are alphabetical,
                False otherwise.
        """
        if word_i.isalpha() and word_j.isalpha():
            return True
        return False

    @staticmethod
    def has_relevant_tag(bigram, relevant):
        """Checks if a bigram consists of at least one relevant tag.

        If iterable of relevant tags is empty, always returns True.

        Args:
            bigram:
                Iterable of two strings
            relevant:
                Iterbale of strings, representing valid tags
                used by Penn Treebank.

        Returns:
            True if intersection between tagged bigram and relevant tags is
            at least one or if relevant tags are empty. False otherwise.
        """
        relevant = set(relevant)
        tags = {tag for word, tag in pos_tag(bigram)}
        if relevant.intersection(tags) or len(relevant) == 0:
            return True
        return False

    def candidates(self, min_count, stops=None, tags={"NN", "NNS", "NNP"}):
        """
        Generate a list of possible candidates for terminology extraction.

        A bigram is considered a candidate if it has a minimum
        absolute frequency in corpus, only contains alpha-numeric characters,
        doesn't contain tokens in stopword list and consists
        of at least one relevant tag.

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
                # Make sure bigrams are alphabetical.
                if self.is_lexical(word_i, word_j):
                    # Filter out infrequent bigrams.
                    if self.bigrams()[word_i, word_j] >= min_count:
                        if self.has_relevant_tag((word_i, word_j), tags):
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

    def write_candidates_file(self, min_count, stops, tags, filename):
        """Write a file with candidates.

        Each line in the output file will contain one candidate.

        Args:
            min_count (int):
                Minimum frequency a bigram has to have to be considered a
                candidate. Absolute frequency are used.
            stops (list):
                List of strings. If a bigram contains a word of that list, it
                is not considered a candidate. If default is used,
                an empty list is used. Default is None.
            tags:
                Iterbale of strings, representing valid tags
                used by Penn Treebank.
            filename (str):
                The name of the output file.

        Returns:
            None.
        """
        filename = os.path.join(filename)
        candidates = self.candidates(min_count, stops, tags)
        with open(filename, "w", encoding="utf-8") as file:
            for wordi, wordj in candidates:
                file.write("{} {}\n".format(wordi, wordj))
        print("Success: Candidates written to '{}'".format(filename))

    @classmethod
    def demo(cls):
        """A demo for important methods of Preprocess class."""
        print("\tDemo for class Preprocess\n"
              "For each method, you can see its arguments and output. "
              "For more information use the help function.\n\n"
              "Arguments used for instanciating the class:\n"
              "\tcorpus - {}".format(cls.DEMO["corpus"]))
        pre = cls(**cls.DEMO)
        print("{:=^90}".format("corpus_stats()"))
        pre.corpus_stats()
        print("{:=^90}".format("bigrams()"))
        print(pre.bigrams())
        print("{:=^90}".format("bigrams('domain1.txt')"))
        print(pre.bigrams("domain1.txt"))
        print("{:=^90}".format("get_frequency"
                               "([('computational', 'linguistics'), "
                               "('not', 'present')])"))
        print(pre.get_frequency([('computational', 'linguistics'),
                                 ('not', 'present')]))
        print("{:=^90}".format("is_lexical('hello', 'world')"))
        print(pre.is_lexical('hello', 'world'))
        print("{:=^90}".format("is_lexical('hello', '?')"))
        print(pre.is_lexical('hello', '?'))
        print("{:=^90}".format("has_relevant_tag(('computational', "
                               "'linguistics'), "
                               "relevant={'NN', 'NNP', 'NNS'})"))
        print(pre.has_relevant_tag(('computational', 'linguistics'),
                                   relevant={'NN', 'NNP', 'NNS'}))
        print("{:=^90}".format("has_relevant_tag(('is', 'difficult'),"
                               "relevant={'NN', 'NNP', 'NNS'})"))
        print(pre.has_relevant_tag(('is', 'difficult'),
                                   relevant={'NN', 'NNP', 'NNS'}))
        print("{:=^90}".format("candidates(min_count=1, "
                               "stops=['is', 'the', 'for', 'of'], "
                               "tags={'NN', 'NNP', 'NNS'})"))
        print(pre.candidates(min_count=1,
                             stops=['is', 'the', 'for', 'of'],
                             tags={'NN', 'NNP', 'NNS'}))


if __name__ == "__main__":
    Preprocess.demo()
