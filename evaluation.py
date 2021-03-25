# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Evaluation of a set of extracted terms.
"""
import csv
import os


class Evaluation:

    DEMO = {"terms": {('machine', 'translation'): 0.8,
                      ('computational', 'linguistics'): 0.6,
                      ('use', 'machine'): 0.5},
            "golds": {('machine', 'translation'),
                      ('speech', 'recognition')}}

    """
    A class that evaluates a set of extracted terms.

    Attributes:
        terms (dict):
            A dict of extraced bigrams (two-tuples of strings), values are
            the value of the decision function.
        golds (set):
            A set of gold standard bigrams (two-tuples of strings)
        correct_terms (set):
            Intersection of terms and golds.

    Methods:
        precision():
            Number of correct terms divided by number of extracted terms.
        recall():
            Number of correct terms divided by number of gold terms.
        f1():
            Harmonic medium of precision and recall.
        highest_scored(n=100):
            Return the n highest scored terms.
        lowest_scored(n=100):
            Return the n lowest scorede terms.
        from_file():
            Read extracted terms and gold terms from a file.
        demo():
            Get a demo of important methods.
    """

    def __init__(self, terms, golds):
        """
        Construct an instance of Evaluation class.

        Args:
            terms (dict):
                Keys are bigrams (two-tuples of strings), value should be
                value of decision function.
            golds:
                Iterable of bigrams (two-tuples of strings) that are
                considered the standard.

        Returns:
            None.
        """
        self.terms = terms
        self.golds = set(golds)
        if not self.golds:
            raise ValueError("Gold standard must contain at least one element")
        self.correct_terms = set(self.terms).intersection(self.golds)

    def precision(self):
        """Compute precision by dividing number of correct terms by
        number of extracted terms.

        Returns:
            Precision value (float)
        """
        if len(self.terms) == 0:
            return 0
        return len(self.correct_terms) / len(self.terms)

    def recall(self):
        """Compute recall by dividing number of correct terms by
        number of gold standard terms.

        Returns:
            Recall value (float)
        """
        return len(self.correct_terms) / len(self.golds)

    def f1(self):
        """Returns harmonic medium of recall and precision value."""
        prec = self.precision()
        rec = self.recall()
        if not prec and not rec:
            return 0
        return (2 * prec * rec) / (prec + rec)

    def highest_scored(self, n=100):
        """Returns n highest scored terms according to
        self.terms values of bigrams.

        Args:
            n (int):
                Number of terms that should be returned at most.

        Returns:
            list:
                list of max. n terms sorted by their decision value in
                descending order.
        """
        sorted_terms = sorted(self.terms,
                              key=lambda x: self.terms[x],
                              reverse=True)
        return sorted_terms[:n]

    def lowest_scored(self, n=100):
        """Returns n lowest scored terms according to
        self.terms values of bigrams.

        Args:
            n (int):
                Number of terms that should be returned at most.

        Returns:
            list:
                list of max. n terms sorted by their decision value in
                ascending order.
        """
        sorted_terms = sorted(self.terms,
                              key=lambda x: self.terms[x])
        return sorted_terms[:n]

    @classmethod
    def demo(cls):
        """Demo of methods in Evaluation class."""
        print("\tDemo for class Evaluation\n"
              "For each method, you can see its arguments and output. "
              "For more information use the help function.\n\n"
              "Arguments used for instanciating the class:\n"
              "\tExtracted terms - {}\n"
              "\tGold terms - {}".format(cls.DEMO["terms"], cls.DEMO["golds"]))
        eva = cls(**cls.DEMO)
        print("{:=^90}".format("recall()"))
        print(eva.recall())
        print("{:=^90}".format("precision()"))
        print(eva.precision())
        print("{:=^90}".format("f1()"))
        print(eva.f1())
        print("{:=^90}".format("highest_scored(n=1)"))
        print(eva.highest_scored(n=1))
        print("{:=^90}".format("lowest_scored(n=1)"))
        print(eva.lowest_scored(n=1))

    @classmethod
    def from_file(cls, goldfile, extractedfile, ignore=2):
        """Get gold terms and extracted terms from files.

        Skips first lines in extractedfile and only reads in
        well formed lines.

        Args:
            goldfile (str):
                Name of a file with gold standard bigrams.Should contain
                two words seperated by a space in each line.
            extractedfile (str):
                Name of a file with extracted terms and value of
                decision function.First two lines will be ignored.
                After that lines should have the format
                <word> <word>;<value>;<True/False>
                where <word> <word> represents a bigram,
                <value> the value of decision function and
                <True/False> wether or not a bigram is considered
                terminology.
            ignore (int):
                Number of lines at beginning of extractedfile
                that will be skipped.
                Default is 2, because these lines contain values
                for alpha and theta.

        Raises:
            ValueError:
                If lines in extractedfile after first ignored
                lines are malformed.

        Returns:
            Evaluation object
        """
        goldfile = os.path.join(goldfile)
        extractedfile = os.path.join(extractedfile)
        golds = set()
        extracted = dict()
        # Read gold standard terms from file.
        with open(goldfile) as goldfile:
            for line in goldfile:
                line = line.rstrip().split()
                golds.add(tuple(line))
        # Read extracted terms from file.
        with open(extractedfile) as extractedfile:
            csv_reader = csv.reader(extractedfile, delimiter=";")
            line_count = 0
            for line in csv_reader:
                if line_count not in range(ignore):
                    try:
                        bigram, value, isterm = (tuple(line[0].split()),
                                                 float(line[1]),
                                                 line[2])
                        if isterm != "True" and isterm != "False":
                            raise ValueError
                    # Malformed lines.
                    except (IndexError, ValueError):
                        raise ValueError("Malformed input file. "
                                         "The first {} lines "
                                         "are ignored.\n"
                                         "Every line after"
                                         "should have the format: "
                                         "<term>;<float>;"
                                         "<True/False>".format(ignore))
                    if isterm == "True":
                        extracted[bigram] = value
                line_count += 1
        return cls(extracted, golds)


if __name__ == "__main__":
    Evaluation.demo()
