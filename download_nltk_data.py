import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK verilerini indir
print("NLTK verisi indiriliyor...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("İndirme tamamlandı.")