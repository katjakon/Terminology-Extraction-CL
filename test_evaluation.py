# -*- coding: utf-8 -*-
"""
Tests for evaluation class.
"""
import unittest
import os

from evaluation import Evaluation


class TestCaseEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eval = Evaluation(terms={('machine', 'translation'): 0.8,
                                     ('computational', 'linguistics'): 0.6,
                                     ('use', 'machine'): 0.5},
                              golds={('machine', 'translation'),
                                     ('speech', 'recognition')})
        cls.terms_file = "demo/demo_out.csv"
        cls.gold_file = "demo/demo_gold.txt"

    def test_recall(self):
        self.assertEqual(self.eval.recall(), 0.5)

    def test_precisison(self):
        self.assertAlmostEqual(self.eval.precision(), 0.333, places=3)

    def test_fscore(self):
        self.assertEqual(self.eval.f1(), 0.4)

    def test_highest_scored_default(self):
        self.assertListEqual(self.eval.highest_scored(),
                             [('machine', 'translation'),
                              ('computational', 'linguistics'),
                              ('use', 'machine')])

    def test_highest_scored_n_defined(self):
        self.assertEqual(len(self.eval.highest_scored(n=1)),
                         1)

    def test_lowest_scored_default(self):
        self.assertListEqual(self.eval.lowest_scored(),
                             [('use', 'machine'),
                              ('computational', 'linguistics'),
                              ('machine', 'translation')])

    def test_lowest_scored_n_defined(self):
        self.assertEqual(len(self.eval.lowest_scored(n=1)),
                         1)

    def test_correct_terms_type(self):
        self.assertIsInstance(self.eval.correct_terms, set)

    def test_correct_terms_content(self):
        self.assertSetEqual(self.eval.correct_terms,
                            {('machine', 'translation')})

    def test_from_file_first_lines_not_in_terms(self):
        testfile = "demo/demo_out.csv"
        test_gold = "demo/demo_gold.txt"
        eval_file = Evaluation.from_file(test_gold, testfile)
        self.assertNotIn(("alpha",), eval_file.terms)
        self.assertNotIn(("theta",), eval_file.terms)

    def test_from_file_terms_identified(self):
        eval_file = Evaluation.from_file(self.gold_file, self.terms_file)
        self.assertIn(("computational", "linguistics"), eval_file.terms)

    def test_from_file_false_terms_left_out(self):
        eval_file = Evaluation.from_file(self.gold_file, self.terms_file)
        self.assertNotIn(("used", "machine"), eval_file.terms)

    def test_from_file_gold_terms(self):
        eval_file = Evaluation.from_file(self.gold_file, self.terms_file)
        self.assertSetEqual({('speech', 'recognition'),
                             ('machine', 'translation')},
                            eval_file.golds)

    def test_from_file_error_malformed(self):
        temp = "test_malformed.csv"
        with open(temp, "w", encoding="utf-8") as tempfile:
            for i in range(5):
                tempfile.write("a;malformed;line\n")
        self.assertRaises(ValueError,
                          Evaluation.from_file,
                          extractedfile=temp,
                          goldfile=self.gold_file)
        os.remove(temp)


if __name__ == "__main__":
    unittest.main(buffer=True)
