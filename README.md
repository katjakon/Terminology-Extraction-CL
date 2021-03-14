# Terminology-Extraction-CL

## Description
This program extracts bigrams that are relevant terminology for a domain from a given corpus.

## Requirements
All files should be utf-8 encoded.

+ Python 3.8.5
+ NLTK (Natural Language Tool KIT) - See installation instructions [here](https://www.nltk.org/install.html)
+ Reuters Corpus - Import nltk, then run `nltk.download('reuters')`, see [here](http://www.nltk.org/data.html) for more information
+ NLTK's Averaged Perceptron Tagger - Import nltk, then run `nltk.download('averaged_perceptron_tagger')`, see [here](http://www.nltk.org/data.html) for more information
+ NLTK's Word Punctuation Tokenizer - Import nltk, then run `nltk.download('punkt')`,  see [here](http://www.nltk.org/data.html) for more information

## How To Use
Move the domain corpus (standard: acl_texts) to this directory. The corpus should be a directory of text files. 

### Generate Candidates
You can find a predefined list of candidates in the file `preprocess/candidates1.txt`.<br>

To generate your own list run:
`main.py candidates [--stops <stopword file>] [--min_count <integer>] <domain dir> <output file> [<tag> [<tag> ...]]`<br>
Explanation:
+ `--stops <stopword file>`: A file with stopwords that are not allowed to occur in a candidate. Bigrams that contain a word from this file are filtered out.
+ `--min_count <integer>`: The minimum absolute frequency a bigram has to have to be considered a candidate. The default is 4.
+ `<domain dir>`: The directory of the domain corpus.
+ `<output file>`: The name for your output file containing the candidates.
+ `[<tag> [<tag> ...]]`: Any number of Penn Treebank Tags. A tagged bigram needs to contain at least one of these tags to be considered a candidate.<br>

To reproduce the candidates in `preprocess/candidates1.txt` run:<br>
`python main.py candidates --stops preprocess\stops_en.txt --min_count 4 acl_texts\ <your file name> NNS NN NNP`

### Extract Terminology

### Evaluate Extracted Terms

## Examples

## Author
Katja Konermann
Matrikelnummer: 802658
Ein Projekt für die Veranstaltung _Computerlinguistische Techniken_ im Wintersemester 20/21