# -*- coding: utf-8 -*-
"""
Unittests for classes
"""

import unittest

from evaluation import Evaluation
from preprocess import Preprocess
from extraction import Terminology


class TestCaseEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eval = Evaluation(terms={('machine', 'translation'): 0.8,
                                     ('computational', 'linguistics'): 0.6,
                                     ('use', 'machine'): 0.5},
                              golds={('machine', 'translation'),
                                     ('speech', 'recognition')})
        cls.file = "demo/demo_out.csv"

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

    def test_from_file(self):
        pass


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

    # def test_candidates_stopwords(self):
    #     cand1 = self.process.candidates(min_count=1, stops=["our"])
    #     self.assertNotIn(("our", "language"), cand1)
    #     cand2 = self.process.candidates(min_count=1)
    #     self.assertIn(("our", "language"), cand2)

    # def test_candidates_min_count(self):
    #     cand = list(self.process.candidates(min_count=3))
    #     self.assertListEqual(cand, [self.bigram1])


class TestCaseTerminology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bigr_all_domain = ("computational", "linguistics") # once in all files in domain
        cls.bigr_one_domain = ("speech", "recognition") # once in one file
        cls.bigr_some_domain = ("text", "mining") # twice in one file, once in another file.
        cls.bigr_just_domain = ("computational", "linguistics")
        cls.bigr_more_domain = ("machine", "learning")
        cls.bigr_less_domain = ("language", "learning")
        cls.term_obj = Terminology(domain="demo/domain/",
                                   reference="demo/reference/",
                                   candidates={
                                       cls.bigr_all_domain,
                                       cls.bigr_one_domain,
                                       cls.bigr_some_domain,
                                       cls.bigr_just_domain,
                                       cls.bigr_more_domain,
                                       cls.bigr_less_domain
                                       }
                                   )

    def test_domain_relevance_only_in_domain(self):
        relevance = self.term_obj.domain_relevance
        self.assertEqual(relevance[self.bigr_just_domain], 1)

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

    def test_domain_consensus_in_all_files(self):
        consensus = self.term_obj.domain_consensus
        self.assertAlmostEqual(consensus[self.bigr_all_domain],
                               1.098612289,
                               places=5)

    def test_domain_consensus_in_one_file(self):
        consensus = self.term_obj.domain_consensus
        self.assertEqual(consensus[self.bigr_one_domain], 0)

    def test_domain_consensus_in_some_files(self):
        consensus = self.term_obj.domain_consensus
        self.assertAlmostEqual(consensus[self.bigr_some_domain],
                               0.6365141683,
                               places=5)

    def test_weighted_candidates_alpha_05(self):
        pass


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCaseTerminology))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
