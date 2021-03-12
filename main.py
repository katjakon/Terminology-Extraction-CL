# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Main file - Implements command line arguments with argparse.
"""
import argparse
import os

from nltk.corpus import reuters

from extraction import Terminology
from evaluation import Evaluation
from preprocess import Preprocess


def terms_from_file(file, ngram=2):
    terms = set()
    with open(file, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip().split("\t")
            if line:
                term = line[0].split()
                if len(term) == ngram:
                    if ngram == 1:
                        terms.add(*term)
                    else:
                        terms.add(tuple(term))
    return terms


def extract(domain, reference, candidates, output, alpha, theta):
    """
    Extract terminolgy from domain corpus by reading candidates from
    a file and writing terminology to a output file.
    First two lines in output will be values for alpha and theta.
    After that line have the format: <word_i> <word_j>\t<value>

    Args:
        domain:
            Corpus with domain specific content.
            Can either be the name of directory with txt files or
            a nltk corpus object.
        reference :
            Corpus with neutral content.
            Can either be the name of directory with txt files or
            a nltk corpus object.
        candidates (str):
            Path to a file where possible candidates are stored.
            Lines should have the format: <word_i> <word_j>
        output:
            Name of a file where output will be stored.
        alpha (float):
            Value for alpha, weights relevance and consenus
        theta (float):
            Value for theta, threshold for terminology

    Raises:
        OSError:
            If output file already exists.

    Returns:
        None.

    """
    output = os.path.join(output)
    # Check if output exitst to avoid overwriting results.
    if os.path.exists(output):
        raise OSError("Output file already exists.")
    # Read candidates from file.
    cand = terms_from_file(candidates)
    # Extract terminology.
    print("Processing domain and reference corpus...")
    term_obj = Terminology(domain, reference, cand)
    terms = term_obj.terminology(alpha, theta)
    # Write output file.
    with open(output, "w", encoding="utf-8") as out:
        out.write("alpha\t{}\n".format(alpha))
        out.write("theta\t{}\n".format(theta))
        for word_i, word_j in terms:
            out.write("{} {}\t{}\n".format(word_i,
                                           word_j,
                                           terms[(word_i, word_j)]))
    # Success message.
    print("Terminology written to {}".format(output))


def evaluate(extracted, gold):
    """
    Prints precision, recall and f1-score for a file
    with extraced terms.

    Args:
        extracted (str):
            Path to a file where extracted terms are stored. First two lines
            will be ignored, because alpha and theta are stored here.
            After that each line should have the format
            <word_i> <word_j>\t<value>
        gold (str):
            Path to a file where gold standard terms are stored. Each line
            should have the format: <word_i> <word_j>

    Returns:
        None.
    """
    extr_terms = terms_from_file(extracted)
    gold_terms = terms_from_file(gold)
    eval_extrac = Evaluation(extr_terms, gold_terms)
    # Print evaluation metrics.
    print("Recall: {:.3f}".format(eval_extrac.recall()))
    print("Precision: {:.3f}".format(eval_extrac.precision()))
    print("F1-Score: {:.3f}".format(eval_extrac.f1()))


def candidates(corpus, stops, min_count, file):
    """Generates candidates from corpus and writes them to file.

    corpus (str):
        directory with text files.
    min_count (int):
        minimum frequency for a term to be considered a candidate
    file (str):
        name of file where candidates should be stored.
    """
    stops = terms_from_file(stops, ngram=1)
    file = os.path.join(file)
    if os.path.exists(file):
        raise OSError("Output file already exists")
    print("Processing corpus...")
    process = Preprocess(corpus)
    print("Generating candidates...")
    cand = process.candidates(min_count=min_count, stops=stops)
    with open(file, "w", encoding="utf-8") as out:
        for word_i, word_j in cand:
            out.write("{} {}\n".format(word_i, word_j))
    print("Candidates written to {}".format(file))


def main():
    domain = "acl_texts"
    reference = reuters
    cand_file = "preprocess/candidates1.txt"
    gold_file = "gold_terminology.txt"
    stopwords = "nltk_stops_en.txt"
    parser = argparse.ArgumentParser(description="Extract Terminology for "
                                     "Computational Linguistics")
    sub = parser.add_subparsers(dest="command")
    # Add necessary arguments for extraction command.
    extract_pars = sub.add_parser("extract",
                                  help="Extract terminology for a domain")
    extract_pars.add_argument("--dom", default=domain,
                              help="Directory of domain corpus with txt files")
    extract_pars.add_argument("--ref", default=reference,
                              help="Directory of reference corpus "
                              "with txt files")
    extract_pars.add_argument("--cand", default=cand_file,
                              help="File with candidates.")
    extract_pars.add_argument("alpha", type=float,
                              help="Value for weighing consensus "
                              "and relevance")
    extract_pars.add_argument("theta", type=float,
                              help="Threshold when extracting terminology")
    extract_pars.add_argument("out", help="Name for output file")
    # Add necessary arguments for evaluate command.
    eval_pars = sub.add_parser("evaluate",
                               help="Evaluate extracted terminology")
    eval_pars.add_argument("extracted",
                           help="Name of file with extracted terms.")
    eval_pars.add_argument("--gold", default=gold_file,
                           help="Name of file with gold standard terms.")
    # Add necessary arguments for candidates command
    cand_pars = sub.add_parser("candidates",
                               help="Generate possible candidates "
                               "for a domain")
    cand_pars.add_argument("--corpus", default=domain,
                           help="Directory with txt files "
                           "to extract candidates from")
    cand_pars.add_argument("--stops", default=stopwords,
                           help="File with stopwords")
    cand_pars.add_argument("--min_count", default=4, type=int,
                           help="Minimum count for terms "
                           "to be considered candidate")
    cand_pars.add_argument("output", help="Name for the output file.")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            extract(args.dom,
                    args.ref,
                    args.cand,
                    args.out,
                    args.alpha,
                    args.theta)
        elif args.command == "evaluate":
            evaluate(args.extracted, args.gold)
        elif args.command == "candidates":
            candidates(args.corpus,
                       args.stops,
                       args.min_count,
                       args.output)
        else:
            parser.print_help()
    except (OSError, ValueError) as err:
        print("Failure: {}".format(err))
        parser.print_help()


if __name__ == "__main__":
    main()
