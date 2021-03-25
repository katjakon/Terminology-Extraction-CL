# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Unittests for the Terminology class
"""
import math
import os
import unittest

from terminology import Terminology


class TestCaseTerminology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Once in all files in domain and only in domain.
        cls.bigr_equally_only_domain = ("computational", "linguistics")
        # Once in one file in domain corpus.
        cls.bigr_one_domain = ("speech", "recognition")
        # Twice in one file in domain, once in another file of domain.
        cls.bigr_not_equally = ("text", "mining")
        # Twice in domain, once in reference.
        cls.bigr_more_domain = ("machine", "learning")
        # Three times in reference, once in domain.
        cls.bigr_less_domain = ("language", "learning")
        cls.term_obj = Terminology(domain="demo/domain/",
                                   reference="demo/reference/",
                                   candidates={
                                       cls.bigr_equally_only_domain,
                                       cls.bigr_one_domain,
                                       cls.bigr_not_equally,
                                       cls.bigr_more_domain,
                                       cls.bigr_less_domain
                                       }
                                   )

    def test_domain_relevance_between_0_and_1(self):
        relevance = self.term_obj.domain_relevance
        for term in relevance:
            self.assertTrue(relevance[term] <= 1 and relevance[term] >= 0)

    def test_domain_consensus_positive(self):
        consensus = self.term_obj.domain_consensus
        for term in consensus:
            self.assertTrue(consensus[term] >= 0)

    def test_domain_consensus_upper_limit(self):
        consensus = self.term_obj.domain_consensus
        file_number = len(self.term_obj.domain.corpus.fileids())
        for term in consensus:
            self.assertTrue(consensus[term] <= math.log(file_number))

    def test_domain_relevance_only_in_domain(self):
        relevance = self.term_obj.domain_relevance
        self.assertEqual(relevance[self.bigr_equally_only_domain], 1)

    def test_domain_relevance_more_in_domain(self):
        relevance = self.term_obj.domain_relevance
        self.assertAlmostEqual(relevance[self.bigr_more_domain],
                               6/11,
                               places=5)

    def test_domain_relevance_less_in_domain(self):
        relevance = self.term_obj.domain_relevance
        self.assertAlmostEqual(relevance[self.bigr_less_domain],
                               1/6,
                               places=5)

    def test_domain_consensus_equally_distributed(self):
        consensus = self.term_obj.domain_consensus
        self.assertAlmostEqual(consensus[self.bigr_equally_only_domain],
                               1.098612289,
                               places=5)

    def test_domain_consensus_in_one_file(self):
        consensus = self.term_obj.domain_consensus
        self.assertEqual(consensus[self.bigr_one_domain], 0)

    def test_domain_consensus_not_equally_distributed(self):
        consensus = self.term_obj.domain_consensus
        self.assertAlmostEqual(consensus[self.bigr_not_equally],
                               0.6365141683,
                               places=5)

    def test_weigh_candidates_error_alpha_above_one(self):
        weighted = self.term_obj.weigh_candidates
        self.assertRaises(ValueError, weighted, alpha=2)

    def test_weigh_candidates_error_alpha_negative(self):
        weighted = self.term_obj.weigh_candidates
        self.assertRaises(ValueError, weighted, alpha=-2)

    def test_weighted_candidates_alpha_05(self):
        weighted = self.term_obj.weigh_candidates(alpha=0.5)
        self.assertAlmostEqual(weighted[self.bigr_equally_only_domain],
                               1.049305,
                               places=3)

    def test_weighted_candidates_alpha_1(self):
        weighted = self.term_obj.weigh_candidates(alpha=1)
        self.assertAlmostEqual(weighted[self.bigr_equally_only_domain],
                               1,
                               places=3)

    def test_weighted_candidates_alpha_0(self):
        weighted = self.term_obj.weigh_candidates(alpha=0)
        self.assertAlmostEqual(weighted[self.bigr_equally_only_domain],
                               1.0986,
                               places=3)

    def test_extract_terminology_raises_error(self):
        weighted = {("computational", "linguistics"): 1,
                    ("language", "learning"): 0.4}
        self.assertRaises(ValueError,
                          self.term_obj.extract_terminology,
                          theta=-1,
                          weighted_candidates=weighted)

    def test_extract_terminology_treshold(self):
        weighted = {("computational", "linguistics"): 1,
                    ("language", "learning"): 0.4}
        terms = self.term_obj.extract_terminology(theta=0.5,
                                                  weighted_candidates=weighted)
        self.assertNotIn(("language", "learning"), terms)
        self.assertIn(("computational", "linguistics"), terms)

    def test_write_csv_file_exists(self):
        testfile = "test.csv"
        self.term_obj.write_csv(alpha=0.5, theta=1, filename=testfile)
        self.assertTrue(os.path.isfile(testfile))
        os.remove(testfile)

    def test_write_csv_file_first_line(self):
        testfile = "test_firstline.csv"
        self.term_obj.write_csv(alpha=0.5, theta=1, filename=testfile)
        with open(testfile, encoding="utf-8") as file:
            first = file.readline().rstrip()
            self.assertEqual(first, "alpha;0.5")
        os.remove(testfile)

    def test_write_csv_second_line(self):
        testfile = "test_secondline.csv"
        self.term_obj.write_csv(alpha=0.5, theta=1, filename=testfile)
        with open(testfile, encoding="utf-8") as file:
            file.readline()
            second = file.readline().rstrip()
            self.assertEqual(second, "theta;1")
        os.remove(testfile)

    def test_write_csv_three_columns(self):
        testfile = "test_columns.csv"
        self.term_obj.write_csv(alpha=0.5, theta=1, filename=testfile)
        with open(testfile, encoding="utf-8") as file:
            for i in range(2):
                file.readline()
            for line in file:
                line = line.split(";")
                self.assertEqual(len(line), 3)
        os.remove(testfile)


if __name__ == "__main__":
    unittest.main(buffer=True)
