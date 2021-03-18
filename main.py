# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Main file - Implements command line arguments with argparse.

Extract - Class for the command to extract terminology
Evaluate - Class for the command to evaluate extracted terms.
Candidates - Class for the command to generate candidates.
"""
import argparse
import sys
import os

from extraction import Terminology
from evaluation import Evaluation
from preprocess import Preprocess


class Extract:
    """
    A class that extracts terminology from a corpus and
    writes results to a file according to a list of arguments.

    Attributes:
        domain (str):
            Corpus with domain specific content.
            Name of directory with txt files.
        candidates (str):
            Path to a file where possible candidates are stored.
            Lines should have the format: <word_i> <word_j>
        output (str):
            Name of a file where output will be stored.
        alpha (float):
            Value for alpha, weights relevance and consensus
        theta (float):
            Value for theta, threshold for terminology

    Methods:
        read_from_file(file, n=2):
            Read in terms from a file.
        run():
            Extract terminology from domain corpus
            and write results to output file.
    """
    from nltk.corpus import reuters

    REF = reuters

    def __init__(self, sysargs):
        """Instanciate an Extract object

        Args:
            sysargs (list):
                A list of command line arguments.
        """
        self.args = self._parser(sysargs)
        self.corpus = self.args.corpus
        self.candidates = self.read_from_file(self.args.candidates)
        self.out = self.args.out
        self.theta = self.args.theta
        self.alpha = self.args.alpha

    def _parser(self, sysargs):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser("Extract terminology for a domain")
        parser.add_argument("corpus",
                            help="Directory of domain corpus with txt files")
        parser.add_argument("candidates",
                            help="File with candidates.")
        parser.add_argument("-a", "--alpha", type=float,
                            help="Value for weighing consensus "
                            "and relevance",
                            required=True)
        parser.add_argument("-t", "--theta", type=float,
                            help="Threshold when extracting terminology",
                            required=True)
        parser.add_argument("out", help="Name for output file")
        return parser.parse_args(sysargs)

    @staticmethod
    def read_from_file(file, n=2):
        """Read terms from file."""
        terms = set()
        with open(file, encoding="utf-8") as file:
            for line in file:
                line = line.rstrip().split("\t")
                if line:
                    term = line[0].split()
                    if len(term) == n:
                        if n == 1:
                            terms.add(*term)
                        else:
                            terms.add(tuple(term))
        return terms

    def run(self):
        """Extract candidates and write them to the output file.

        Returns: None
        """
        out = os.path.join(self.out)
        # Extract terminology.
        print("Processing domain and reference corpus...")
        term_obj = Terminology(self.corpus,
                               self.REF,
                               self.candidates)
        print("Extracting Terminology...")
        term_obj.write_csv(self.alpha, self.theta, out)


class Evaluate:
    """
    A class that evaluates extracted terms.

    Attributes:
        gold (str):
            Name of a file with gold standard bigrams.
            Line format <word> <word>.
        extracted (str):
            Name of file with extracted terms and values
            of decision function. Line format <word> <word>\t<value>
        high (int):
            Indicates how many of the highest scored terms will be
            printed. If None, no terms will be printed.
        low (int):
            Indicates how many of the lowest scored terms will be
            printes, If None, no terms will be printed.
    """

    def __init__(self, sysargs):
        """Instanciating an Evaluate object.

        Args:
            sysargs (list): Command Line arguments.

        Returns:
            None
        """
        self._args = self._parser(sysargs)
        self.gold = self._args.gold
        self.extracted = self._args.extracted
        self.high = self._args.high
        self.low = self._args.low

    def _parser(self, sysargs):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Evaluate "
                                         "extracted terminology")
        parser.add_argument("--extracted",
                            help="Name of file with extracted terms.",
                            required=True)
        parser.add_argument("--gold",
                            help="Name of file with gold standard terms.",
                            required=True)
        parser.add_argument("--high", type=int,
                            help="Print n highest scored terms")
        parser.add_argument("--low", type=int,
                            help="Print n lowest scored terms")
        return parser.parse_args(sysargs)

    def run(self):
        """Evaluate extracted terms and print highest/lowest scored terms."""
        eval_extrac = Evaluation.from_file(self.gold, self.extracted)
        # Print evaluation metrics.
        print("Recall: {:.3f}".format(eval_extrac.recall()))
        print("Precision: {:.3f}".format(eval_extrac.precision()))
        print("F1-Score: {:.3f}".format(eval_extrac.f1()))
        if self.high is not None:
            print("{} highest scored terms:".format(self.high))
            high_terms = eval_extrac.highest_scored(self.high)
            for wordi, wordj in high_terms:
                print(wordi, wordj)
        if self.low is not None:
            print("{} lowest scored terms:".format(self.low))
            low_terms = eval_extrac.lowest_scored(self.low)
            for wordi, wordj in low_terms:
                print(wordi, wordj)


class Candidates(Extract):
    """
    A class that creates a file with possible candidates.

    Attributes:
        corpus (str):
            directory with text files.
        min_count (int):
            minimum frequency for a term to be considered a candidate
        output (str):
            name of file where candidates should be stored.
        stops (str):
            Name of a file with stopwords that should be ignored.
            If not defined, None.
        tags [list]:
            List of Penn Treebank Tags that are considered relevant for
            a candidate, can be empty.
    """

    def __init__(self, sysargs):
        self.args = self._parser(sysargs)
        self.corpus = self.args.corpus
        self.stops = self.args.stops
        self.min_count = self.args.min_count
        self.output = self.args.output
        self.tags = self.args.tags

    def _parser(self, sysargs):
        parser = argparse.ArgumentParser(description="Generate possible "
                                         "candidates for a domain")
        parser.add_argument("corpus",
                            help="Directory with txt files "
                            "to extract candidates from")
        parser.add_argument("output", help="Name for the output file.")
        parser.add_argument("--stops",
                            help="File with stopwords")
        parser.add_argument("--min_count", default=4, type=int,
                            help="Minimum count for terms "
                            "to be considered candidate")
        parser.add_argument("tags",
                            help="Relevant tags for candidates, "
                            "use Penn Treebank Tags",
                            nargs="*",
                            default=[])
        return parser.parse_args(sysargs)

    def run(self):
        """Generate candidates and write them to a file

        Returns: None.
        """
        if self.stops is None:
            stops = []
        else:
            stops = self.read_from_file(self.stops, n=1)
        out = os.path.join(self.output)
        print("Processing corpus...")
        process = Preprocess(self.corpus)
        print("Generating candidates...")
        process.write_candidates_file(min_count=self.min_count,
                                      stops=stops,
                                      tags=self.tags,
                                      filename=out)


def main():
    arg = sys.argv
    if len(arg) < 2:
        raise ValueError("Enter a valid command.")
    if arg[1] == "extract":
        Extract(arg[2:]).run()
    elif arg[1] == "evaluate":
        Evaluate(arg[2:]).run()
    elif arg[1] == "candidates":
        Candidates(arg[2:]).run()
    else:
        raise ValueError("Invalid command '{}'".format(arg[1]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as err:
        print("Failure: {}".format(err))
        print("Type 'evaluate -h', 'extract -h' "
              "or 'candidates -h' for information about commands")
