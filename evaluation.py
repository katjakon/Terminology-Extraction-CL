# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Evaluation of a set of extracted terms.
"""
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
        terms (set): A set of extraced bigrams (two-tuples of strings)
        golds (set): A set of gold standard bigrams (two-tuples of strings)
        correct_terms (set): Intersection of terms and golds.

    Methods:
        precision():
            Number of correct terms divided by number of extracted terms.
        recall():
            Number of correct terms divided by number of gold terms.
        f1():
            Harmonic medium of precision and recall.
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
            raise ValueError("Gold standard must contain at least one bigram")
        self.correct_terms = set(self.terms).intersection(self.golds)

    def precision(self):
        """
        Compute precision by dividing number of correct terms by
        number of extracted terms.

        Returns:
            Precision value (float)

        """
        if len(self.terms) == 0:
            return 0
        return len(self.correct_terms) / len(self.terms)

    def recall(self):
        """
        Compute recall by dividing number of correct terms by
        number of gold standard terms.

        Returns:
            Recall value (float)

        """
        return len(self.correct_terms) / len(self.golds)

    def f1(self):
        """Returns harmonic medium of recall and precision value.
        """
        prec = self.precision()
        rec = self.recall()
        if not prec and not rec:
            return 0
        return (2 * prec * rec) / (prec + rec)

    def highest_scored(self, n=100):
        sorted_terms = sorted(self.terms,
                              key=lambda x: self.terms[x],
                              reverse=True)
        return sorted_terms[:n]

    def lowest_scored(self, n=100):
        sorted_terms = sorted(self.terms,
                              key=lambda x: self.terms[x])
        return sorted_terms[:n]

    @classmethod
    def demo(cls):
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
    def from_file(cls, goldfile, extractedfile):
        """Get gold terms and extracted terms from files.

        The gold file should contain two words seperated by a space
        in each line. The file with the extracted terms should have the
        format <word> <word>\t<value> where value is a float that represents
        the value of a decision function.

        Args:
            goldfile (str):
                Name of a file with gold standard bigrams.
            extractedfile (str):
                Name of a file with extracted terms and value of
                decision function.

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
                # Ignore malformed lines.
                if len(line) == 2:
                    golds.add(tuple(line))
        # Read extracted terms from file.
        with open(extractedfile) as extractedfile:
            for line in extractedfile:
                line = line.rstrip().split("\t")
                if len(line) == 2:
                    # Ignore malformed lines.
                    try:
                        bigram, value = line[0].split(), float(line[1])
                        if len(bigram) == 2:
                            extracted[tuple(bigram)] = value
                    except ValueError:
                        pass
        return cls(extracted, golds)


if __name__ == "__main__":
    d = Evaluation.from_file("gold_terminology.txt", "output/output1.txt")
    print(d.f1())