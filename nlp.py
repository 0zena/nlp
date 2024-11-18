import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

from langdetect import detect

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')

class Tasks:
    @staticmethod
    def word_frequency(text, key):
        words = word_tokenize(text.lower())
        return FreqDist(words)[key.lower()]
    # end define
    
    @staticmethod
    def language(text):
        return detect(text)
    # end define

    @staticmethod
    def sentance_compare(text1, text2):
        vectors = CountVectorizer().fit_transform([text1, text2])
        similarity_matrix = cosine_similarity(vectors)
        return similarity_matrix[0, 1] * 100
    # end define

    @staticmethod
    def analyze_sentiment(text):
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0: return "Pozitīvs"
        elif sentiment < 0: return "Negatīvs"
        else: return "Neitrāls"
    # end define

    @staticmethod
    def normalize(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[!?.]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()        
        return text
    # end define

    @staticmethod
    def summarize(text):
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
        word_frequencies = FreqDist(words)
        sentence_scores = {sentence: sum(word_frequencies[word.lower()] for word in word_tokenize(sentence) if word.lower() in word_frequencies) for sentence in sentences}
        summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]
        return " ".join(summarized_sentences)
    # end define

    @staticmethod
    def word_embeddings(word1, word2, word3):
        
    # end define

# end class

    

text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."
raw_text = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"

print(f"Vārdu sakritības līmenis: {Tasks.sentance_compare(text1, text2):.2f}%")
print(Tasks.normalize(raw_text))
article = "Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm."
print(Tasks.summarize(article))

# Teikumi
sentences = [
    "Šis produkts ir lielisks, esmu ļoti apmierināts!",
    "Esmu vīlies, produkts neatbilst aprakstam.",
    "Neitrāls produkts, nekas īpašs."
]

# Noskaņojuma analīze katram teikumam
for sentence in sentences:
    sentiment = Tasks.analyze_sentiment(sentence)
    print(f"Teikums: \"{sentence}\" - Noskaņojums: {sentiment}")


# Ievades raksts
article = """
Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. 
Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. 
Tā ir viena no Eiropas Savienības dalībvalstīm.
"""

# Rezumējam tekstu
summary = Tasks.summarize(article)


# Example usage
word1 = "māja"
word2 = "dzīvoklis"
word3 = "jūra"

most_similar = Tasks.word_embeddings(word1, word2, word3)
print(f"\nThe most semantically similar words are: {most_similar[0]} and {most_similar[1]}")


