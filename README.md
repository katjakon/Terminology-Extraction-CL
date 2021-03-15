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
Move the domain corpus (standard: `acl_texts`) to this directory. The corpus should be a directory of text files. 

### Generate Candidates
To extract terminology for a domain, you have to choose possible candidates first.
A predefined list of candidates can be found in the file `preprocess/candidates1.txt`.<br>

To generate your own list run:<br>
`main.py candidates [--stops <stopword file>] [--min_count <integer>] <domain dir> <output file> [<tag> [<tag> ...]]`<br>
__Explanation:__
+ `--stops <stopword file>`: A file with stopwords that are not allowed to occur in a candidate. Bigrams that contain a word from this file are filtered out. If argument is left out, no stopwords will be used.
+ `--min_count <integer>`: The minimum absolute frequency a bigram has to have to be considered a candidate. The default is 4.
+ `<domain dir>`: The directory of the domain corpus.
+ `<output file>`: The name for your output file containing the candidates.
+ `[<tag> [<tag> ...]]`: Any number of Penn Treebank Tags. A tagged bigram needs to contain at least one of these tags to be considered a candidate. If argument is left out, no tagging will be used.<br>

To reproduce the candidates in `preprocess/candidates1.txt` run:<br>
`python main.py candidates --stops preprocess\stops_en.txt --min_count 4 acl_texts\ <your file name> NNS NN NNP`

### Extract Terminology
Use a file with candidates and the domain corpus to extract relevant terminology. Run: <br>
`main.py extract -a <value for alpha> -t <value for theta> <domain dir> <candidates file> <output file>`<br>
__Explanation:__<br>
+ `-a <value for alpha>`: A float between 0 and 1. Used to weigh domain consensus and domain relevance. If greater than 0.5 domain relevance has more weight, if less than 0.5 domain consenus has more weight.
+ `-t <value for theta>`: A positive float. Used as a threshold when determining terminology.
+ `<domain dir>`: Directory of domain corpus. Standard should be `acl_texts`.
+ `<candidates file>`: A file with candidates, generated by `main.py candidates`.
+ `<output file>` : The name for the output file where extracted terms are stored.

__Example:__<br>
`main.py extract -a 0.5 -t 2 acl_texts preprocess\candidates1.txt output/output1.txt`

### Evaluate Extracted Terms
To evaluate extracted terms run:
`main.py evaluate --extracted <term file> --gold <gold file> [--high <int>] [--low <int>]`<br>
__Explanation:__<br>
+ `--extracted <term file>`: A file with extracted terms, generated by `main.py extract`.
+ `--gold <gold file>`: A file with gold standard terminology. Standard should be `gold_terminology.txt`
+ `--high <int>`: Optionally, define an integer and print out the n highest scored terms.
+ `--low <int>`: Optionally, define an integer and print out the n lowest scored terms.

__Example:__<br>
`main.py evaluate --extracted output\output1.txt --gold gold_terminology.txt --high 30`


## Author
Katja Konermann<br>
Matrikelnummer: 802658<br>
Ein Projekt für die Veranstaltung _Computerlinguistische Techniken_ im Wintersemester 20/21