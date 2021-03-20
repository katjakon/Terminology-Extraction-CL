# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Unittests for the Preprocess class.
"""
import unittest
import os

from preprocess import Preprocess


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

    def test_candidates_with_stopwords(self):
        cand1 = self.process.candidates(min_count=1, stops=["the", "of"])
        self.assertNotIn(("the", "field"), cand1)
        self.assertNotIn(("field", "of"), cand1)

    def test_candidates_without_stopwords(self):
        cand2 = self.process.candidates(min_count=1)
        self.assertIn(("the", "field"), cand2)
        self.assertIn(("field", "of"), cand2)

    def test_candidates_min_count(self):
        cand = self.process.candidates(min_count=3)
        self.assertSetEqual(cand, {self.bigram1, self.bigram3})

    def test_write_candidates_file_exists(self):
        temp = "test_candidates.txt"
        self.process.write_candidates_file(min_count=1,
                                           stops=["the", "of"],
                                           tags=[],
                                           filename=temp)
        self.assertTrue(os.path.exists(temp))
        os.remove(temp)

    def test_write_candidates_file_format(self):
        temp = "test_candidates_format.txt"
        self.process.write_candidates_file(min_count=1,
                                           stops=["the", "of"],
                                           tags=[],
                                           filename=temp)
        with open(temp, encoding="utf-8") as file:
            for line in file:
                line = line.split()
                self.assertEqual(len(line), 2)
        os.remove(temp)


if __name__ == "__main__":
    unittest.main(buffer=True)
