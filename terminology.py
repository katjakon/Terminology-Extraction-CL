# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Extracting terminolgy from a corpus.
"""
import csv
import math
import os

from preprocess import Preprocess


class Terminology:

    DEMO = {"domain": "demo/domain/",
            "reference": "demo/reference",
            "candidates": {('computational', 'linguistics'),
                           ('text', 'mining'),
                           ('language', 'learning'),
                           ('speech', 'recognition'),
                           ('machine', 'learning')}
            }

    """
    A class for extracting terminolgy from a domain corpus.

    Attributes:
        domain:
            a Preprocess object of the domain corpus
        reference:
            a Preprocess object of the reference corpus
        candidates:
            a set of bigrams (two-tuples of str) that could be terminology
        domain_relevance:
            a dict that contains relevance for each term in candidates
        domain_consensus:
            a dict that contains consensus for each term in candidates.

    Methods:
        weigh_candidates(alpha):
            Weighs relevance and consensus by alpha for
            each candidate.
        extract_terminology(theta, weighted_candidates):
            Extracts terms that exceed treshold theta.
        write_csv(alpha, theta, filename):
            Write a csv file with each candidate, its value
            when weighed by alpha and whether its terminology or not.
        demo():
            Get a demo of key methods.
    """

    def __init__(self, domain, reference, candidates):
        """Construct a Terminolgy instance.

        Args:
            domain:
                A corpus with texts from a specific domain.
                Can either be a path to a directory with text files or
                a nltk corpus object.
            reference:
                A corpus with texts from a neutral domain.
                Can either be a path to a directory with text files or
                a nltk corpus object.
            candidates:
                A set of bigrams (two-tuples of strings) that could be
                considered terminology.

        Returns:
            None.
        """
        self.domain = Preprocess(domain)
        self.reference = Preprocess(reference)
        self.candidates = set(candidates)
        self.domain_relevance = self._domain_relevance()
        self.domain_consensus = self._domain_consensus()

    @staticmethod
    def _probability(freq, freq_sum):
        """Returns probabilty by dividing freq by freq_sum"""
        if freq_sum == 0:
            return 0
        return freq / freq_sum

    def _domain_relevance(self):
        """
        Computes domain relevance for each term in self.candidates.

        Domain relevance of term is defined as its probability in domain corpus
        divided by sum of probability in domain corpus and its probability in
        reference corpus.

        Returns:
            dict:
                Keys are bigrams, values are domain relevance of that term.
                1 means bigram only occurs in domain, <0.5 means bigrams
                occurs more often in reference.
        """
        print("Computing domain relevance...")
        domain_relevance = dict()
        # Get frequency of candidates in domain and reference.
        freq_dom = self.domain.get_frequency(self.candidates)
        freq_ref = self.reference.get_frequency(self.candidates)
        # Sum of frequency of all candidate in corpora.
        sum_dom = sum(freq_dom[bigram] for bigram in freq_dom)
        sum_ref = sum(freq_ref[bigram] for bigram in freq_ref)
        for candidate in self.candidates:
            # Get probabilty of a term.
            prob_ref = self._probability(freq_ref.get(candidate, 0), sum_ref)
            prob_dom = self._probability(freq_dom.get(candidate, 0), sum_dom)
            # Set relevance.
            if prob_dom == 0 and prob_ref == 0:
                term_relevance = 0
            else:
                term_relevance = prob_dom / (prob_dom + prob_ref)
            domain_relevance[candidate] = term_relevance
        return domain_relevance

    def _domain_consensus(self):
        """
        Computes domain consensus for every bigram in self.candidates.

        Domain consensus of a term is defined as entropy of
        probabilty of distribution of term over all documents.

        Returns:
            dict:
                keys are the bigrams, values is the domain consensus.
        """
        print("Computing domain consensus...")
        domain_consensus = dict()
        files = {term: dict() for term in self.candidates}
        for file in self.domain.corpus.fileids():
            # Get frequency of candidates in file.
            cand_freq = self.domain.get_frequency(self.candidates, file)
            # For each candidate set frequency in file.
            for term in cand_freq:
                files[term][file] = cand_freq[term]
        for term in files:
            # Sum of frequency of a term.
            sum_files = sum(files[term][file] for file in files[term])
            for file in files[term]:
                # Divide frequency of a term in a file by sum of freq.
                files[term][file] = self._probability(files[term][file],
                                                      sum_files)
            # Compute entropy of distribution for a term.
            cons = sum(files[term][doc] * math.log(1/files[term][doc])
                       if files[term][doc] != 0 else 0
                       for doc in files[term])
            domain_consensus[term] = cons
        return domain_consensus

    def weigh_candidates(self, alpha):
        """
        Extract terminolgy based on domain consensus and
        relevance.

        Arg:
            alpha (float):
                Determines proportion between domain relevance and
                domain consensus. If greater than 0.5 domain relevance
                has more weight, if less than 0.5 domain consenus has more
                weight.

        Raises:
            ValueError:
                If alpha is between 0 and 1.

        Returns:
            dict:
                Keys are candidates, values is value of weighing consensus
                and relevance.
        """
        try:
            # Make sure alpha has appropiate value.
            assert alpha >= 0 and alpha <= 1, "Alpha should range from 0 to 1"
        except AssertionError as err:
            raise ValueError(err)
        terms = dict()
        for candidate in self.candidates:
            # Get value by weighing relevance and consensus of
            # each candidate.
            value = (alpha * self.domain_relevance[candidate]
                     + (1-alpha)*self.domain_consensus[candidate])
            terms[candidate] = value
        return terms

    @staticmethod
    def extract_terminology(theta, weighted_candidates):
        """Determines relevant terminology bases on a threshold
        and weighted candidates.

        Args:
            theta (float):
                A positive float that acts as a threshold
                for determining terminology
            weighted_candidates (dict):
                A dict where the keys are tuples of strings and the values
                are floats.

        Raises:
            ValueError:
                If theta is negative.

        Returns:
            Set of tuples of strings that exceeds the threshold theta.
        """
        try:
            assert theta >= 0, "Theta needs to be positive."
        except AssertionError as err:
            raise ValueError(err)
        return {term for term in weighted_candidates
                if weighted_candidates[term] > theta}

    def write_csv(self, alpha, theta, filename):
        """
        Extract terminolgy based on domain consensus and
        relevance.

        Arg:
            alpha (float):
                Determines proportion between domain relevance and
                domain consensus. If greater than 0.5 domain relevance
                has more weight, if less than 0.5 domain consenus has more
                weight.
            theta: Threshold for candidate to be considered terminology.

        Raises:
            ValueError:
                If alpha is not in range 0,1 or
                if theta is not positive.

        Returns:
            dict:
                Keys are the final extraced terminology, values are value of
                decision function.
        """
        filename = os.path.join(filename)
        weighted = self.weigh_candidates(alpha)
        terms = self.extract_terminology(theta, weighted)
        sort_weighted = sorted(weighted,
                               key=lambda x: weighted[x],
                               reverse=True)
        with open(filename, "w", encoding="utf-8", newline="") as file:
            csv_writer = csv.writer(file, delimiter=";")
            csv_writer.writerow(["alpha", alpha])
            csv_writer.writerow(["theta", theta])
            for wordi, wordj in sort_weighted:
                bigram = "{} {}".format(wordi, wordj)
                value = weighted[wordi, wordj]
                extracted = False
                # Is bigram considered terminology or not.
                if (wordi, wordj) in terms:
                    extracted = True
                csv_writer.writerow([bigram,
                                     value,
                                     extracted])
        print("Success: Terms written to '{}'".format(filename))

    @classmethod
    def demo(cls):
        """Demo for key functionalities of Terminology class"""
        print("\tDemo for class Terminology\n"
              "For each method, you can see its arguments and output. "
              "For more information use the help function.\n\n"
              "Arguments used for instanciating the class:\n"
              "\tDomain corpus - {}\n"
              "\tReference corpus - {}\n"
              "\tCandidates - {}".format(cls.DEMO["domain"],
                                         cls.DEMO["reference"],
                                         cls.DEMO["candidates"]))
        term = cls(**cls.DEMO)
        print("{:=^100}".format("weigh_candidates(alpha=0.5)"))
        print(term.weigh_candidates(alpha=0.5))
        print("{:=^100}".format("extract_terminology(0.5, "
                                "{('computational', 'linguistics'):1,"
                                "('language', 'learning'):0.4 })"))
        weighted = {('computational', 'linguistics'): 1,
                    ('language', 'learning'): 0.4}
        print(term.extract_terminology(theta=0.5,
                                       weighted_candidates=weighted))
        print("{:=^100}".format("write_csv(alpha=0.6, theta=0.5, "
                                "filename='demo/demo_out.csv')"))
        term.write_csv(alpha=0.6, theta=0.5, filename='demo/demo_out.csv')


if __name__ == "__main__":
    Terminology.demo()
