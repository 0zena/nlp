import re
import spacy
import nltk
import torch

from translate import Translator
from transformers import pipeline
from transformers import BertTokenizer, BertModel

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langdetect import detect
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
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
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        def get_embedding(word):
            inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                return model(**inputs).last_hidden_state[0, 0, :].numpy()

        embeddings = {word: get_embedding(word) for word in [word1, word2, word3]}

        similarities = {
            (word1, word2): cosine_similarity([embeddings[word1]], [embeddings[word2]])[0][0],
            (word1, word3): cosine_similarity([embeddings[word1]], [embeddings[word3]])[0][0],
            (word2, word3): cosine_similarity([embeddings[word2]], [embeddings[word3]])[0][0],
        }

        for (w1, w2), sim in similarities.items():
            print(f"Similarity between '{w1}' and '{w2}': {sim:.4f}")

        return
    # end define

    @staticmethod
    def recognize_phrases(text):
        nlp = spacy.load("xx_ent_wiki_sm")
        doc = nlp(text)

        # modelis sūdīgi saprot latviešu valodu
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                print(f"Personvārds: {ent.text}")
            elif ent.label_ == "ORG":
                print(f"Organizācija: {ent.text}")
    #end define

    @staticmethod
    def generate_text(text):
        generator = pipeline("text-generation", model="gpt2")
        text = Translator(from_lang="lv", to_lang="en").translate(text)
        json = generator(
            text,
            max_length=60,
            num_return_sequences=1,
            temperature=1.2,
            truncation=True
        )

        result = Translator(from_lang="en", to_lang="lv").translate(json[0]["generated_text"]) 
        return print(result)
    # end defince

    @staticmethod
    def translate_text(text):
        text = Translator(from_lang="lv", to_lang="en").translate(text)
        return print(text)
    # end define
# end class


class Examples():
    @staticmethod
    def get_frequency():
        print("1.usd\n")
        text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."
        key = "Kaķis"

        return print(f"Frequency of key {key}: {Tasks.word_frequency(text, key)}")
    # end define

    @staticmethod
    def get_language():
        print("2.usd\n")
        texts = [
            "Šodien ir saulaina diena.",
            "Today is a sunny day.",
            "Сегодня солнечный день.",
        ]

        for text in texts:
            print(Tasks.language(text))
    # end define

    @staticmethod
    def get_sentance_compare():
        print("3.usd\n")
        text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
        text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."

        return print(f"Vārdu sakritības līmenis: {Tasks.sentance_compare(text1, text2):.2f}%")
    # end define

    @staticmethod
    def get_analyze_sentiment():
        print("4.usd\n")
        sentences = [
            "Šis produkts ir lielisks, esmu ļoti apmierināts!",
            "Esmu vīlies, produkts neatbilst aprakstam.",
            "Neitrāls produkts, nekas īpašs."
        ]

        for sentence in sentences:
            sentiment = Tasks.analyze_sentiment(sentence)
            print(f"Teikums: \"{sentence}\" - Noskaņojums: {sentiment}")
    # end define

    @staticmethod
    def get_normalize():
        print("5.usd\n")
        text = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"

        return print(Tasks.normalize(text))
    # end define

    @staticmethod
    def get_summarize():
        print("6.usd\n")
        text = """
            Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. 
            Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. 
            Tā ir viena no Eiropas Savienības dalībvalstīm.
            """
        
        return print(Tasks.summarize(text))
    # end define

    @staticmethod
    def get_word_embeddings():
        print("7.usd\n")

        return Tasks.word_embeddings("māja", "dzīvoklis", "jūra")
    # end define

    @staticmethod
    def get_recognize_phrases():
        print("8.usd\n")

        return Tasks.recognize_phrases("Valsts prezidents Egils Levits piedalījās pasākumā, ko organizēja Latvijas Universitāte.")
    # end define

    @staticmethod
    def get_generate_text():
        print("9.usd\n")

        return Tasks.generate_text("Reiz kādā tālā zemē...")
    # end define

    @staticmethod
    def get_translate():
        print("10.usd\n")

        texts = [
            "Labdien! Kā jums klājas?",
            "Es šodien lasīju interesantu grāmatu.",
        ]

        for text in texts:
            Tasks.translate_text(text)
    # end define
# end class

Examples.get_frequency()
Examples.get_language()
Examples.get_sentance_compare()
Examples.get_analyze_sentiment()
Examples.get_normalize()
Examples.get_summarize()
Examples.get_word_embeddings()
Examples.get_recognize_phrases()
Examples.get_generate_text()
Examples.get_translate()
