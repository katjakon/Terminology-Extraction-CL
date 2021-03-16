# -*- coding: utf-8 -*-
"""
Unittests for classes
"""

import unittest

from evaluation import Evaluation
from preprocess import Preprocess


class TestCaseEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eval = Evaluation(terms={('machine', 'translation'): 0.8,
                                     ('computational', 'linguistics'): 0.6,
                                     ('use', 'machine'): 0.5},
                              golds={('machine', 'translation'),
                                     ('speech', 'recognition')})

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


class TestCasePreprocess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.process = Preprocess("demo/domain")
        cls.fileid = "domain1.txt"
        cls.bigram1 = ('computational', 'linguistics')
        cls.bigram2 = ('?/', "the")
        cls.bigram3 = ("text", "mining")

    def test_is_lexical_on_lexical_bigram(self):
        wordi, wordj = self.bigram1
        self.assertTrue(self.process.is_lexical(wordi, wordj))

    def test_is_lexcial_on_non_lexical_bigram(self):
        wordi, wordj = self.bigram2
        self.assertFalse(self.process.is_lexical(wordi, wordj))

    def test_relevant_tags_empty_tagset(self):
        self.assertTrue(self.process.has_relevant_tag(self.bigram3,
                                                      set()))

    def test_relevant_tags_with_tagset(self):
        self.assertTrue(self.process.has_relevant_tag(self.bigram3,
                                                      {"NN"}))

    def test_bigrams_error_nonexisting_fileid(self):
        self.assertRaises(AssertionError,
                          self.process.bigrams,
                          fileid="nonexisting")

    def test_bigrams(self):
        self.assertIn(self.bigram3,
                      self.process.bigrams())

    def test_bigrams_for_file(self):
        self.assertIn(self.bigram1,
                      self.process.bigrams(self.fileid))
        self.assertNotIn(self.bigram3,
                         self.process.bigrams(self.fileid))

    def test_get_frequency_with_nonexisting_bigram(self):
        self.assertNotIn(self.bigram2,
                         self.process.get_frequency([self.bigram2]))

    def test_get_frequency_with_existing_bigram(self):
        self.assertDictEqual(self.process.get_frequency([self.bigram1]),
                             {self.bigram1: 3})

    def test_get_frequency_with_fileid(self):
        self.assertDictEqual(self.process.get_frequency([self.bigram1],
                                                        fileid=self.fileid),
                             {self.bigram1: 1})
if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCaseEvaluation))
    suite.addTest(unittest.makeSuite(TestCasePreprocess))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
