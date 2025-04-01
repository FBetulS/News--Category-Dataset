#!/bin/bash

# NLTK verilerini indir
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

echo "Kurulum tamamlandı!"