import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from pprint import pprint

# SKLearn
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import logging
logging.getLogger("LdaMallet").setLevel(logging.WARNING)


class SurveyNLP(object):
    # This class will query a connected PostgreSQL database using psycopg2, and
    # then clean the data for modeling.

    def __init__(self):
        self.stop_words = text.ENGLISH_STOP_WORDS.union(['vmp',
                                                        'really',
                                                        'like',
                                                        'na',
                                                        'noresponse'])
        self.mallet_path = ("/Users/timothymiller/Galvanize/capstone_2_3"
                            "/mallet-2.0.8/bin/mallet")


    def load_survey_data(self, file_path='2019 Member Survey - Raw Data.csv'):
        self.df = pd.read_csv(file_path, header=[0,1], low_memory=False)
        self.df_noobs = self.df[(self.df.iloc[:,33] == 'I just started') 
                         | (self.df.iloc[:,33] == '6 - 12 months') 
                         | (self.df.iloc[:,33] == '1-3 years')]
        return self.df, self.df_noobs


    def display_sklearn_topics(self, model, feature_names, num_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx}:")
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-num_words - 1:-1]]))

    def sklearn_lda(self, col_num, data_frame, num_topics=5, num_words=15):
        data_frame.iloc[:,col_num].fillna(value='noresponse', inplace=True)
        col_text = list(data_frame.iloc[:,col_num].to_numpy())
        
        sklearn_vect = CountVectorizer(lowercase=True,
                                       tokenizer=None,
                                       stop_words=self.stop_words,
                                       ngram_range=(0,5),
                                       max_features=1000)
        sklearn_vect.fit(col_text)
        sklearn_trans_text = sklearn_vect.fit_transform(col_text)
        sklearn_feature_names = sklearn_vect.get_feature_names()
        sklearn_lda = LatentDirichletAllocation(n_components=num_topics,
                                                max_iter=10,
                                                learning_method='online',
                                                random_state=42,
                                                n_jobs=-1)
        sklearn_lda.fit(sklearn_trans_text)
        label = data_frame.columns.to_numpy()[col_num]
        print(f"Column {col_num} - Label: {label}\n")
        print(f"SKLearn LDA Topic Modeling with {num_topics} topics:\n")
        self.display_sklearn_topics(sklearn_lda,
                                    sklearn_feature_names,
                                    num_words)


    def sklearn_nmf(self, col_num, data_frame, num_topics=5, num_words=15):
        from sklearn.decomposition import NMF
        data_frame.iloc[:,col_num].fillna(value='noresponse', inplace=True)
        col_text = list(data_frame.iloc[:,col_num].to_numpy())
        
        sklearn_vect = CountVectorizer(lowercase=True,
                                       tokenizer=None,
                                       stop_words=self.stop_words,
                                       ngram_range=(0,5),
                                       max_features=1000)
        sklearn_vect.fit(col_text)
        sklearn_trans_text = sklearn_vect.fit_transform(col_text)
        sklearn_feature_names = sklearn_vect.get_feature_names()
        
        
        sklearn_nmf = NMF(n_components=num_topics, init='random', random_state=0)
        
        W = sklearn_nmf.fit_transform(sklearn_trans_text)
        H = pd.DataFrame(sklearn_nmf.components_)
        label = data_frame.columns.to_numpy()[col_num]
        print(f"Column {col_num} - Label: {label}\n")
        print(f"SKLearn NMF Topic Modeling with {num_topics} topics:\n")
        self.display_sklearn_topics(sklearn_nmf,
                                    sklearn_feature_names,
                                    num_words)
        

    def gensim_sentences_to_words(self, sentences):
        # Convert a document into a list of tokens.
        # This lowercases, tokenizes, de-accents (optional).
        # – the output are final tokens = unicode strings,
        # that won’t be processed any further.
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence)))

    def make_bigrams(self, texts):
        # Build bigram model - higher threshold = fewer phrases
        self.bigram = gensim.models.Phrases(texts, min_count=5, threshold=20)
        
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_model = gensim.models.phrases.Phraser(self.bigram)
        
        return [self.bigram_model[doc] for doc in texts]


    def make_trigrams(self, texts):
        # Build trigram model - higher threshold = fewer phrases
        self.trigram = gensim.models.Phrases(self.bigram[texts], threshold=15)  
            
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.trigram_model = gensim.models.phrases.Phraser(self.trigram)

        return [self.trigram_model[self.bigram_model[doc]] for doc in texts]


    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc))
                if word not in self.stop_words] for doc in texts]

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        '''https://spacy.io/api/annotation
        Initialize spacy 'en' model, keeping only tagger component
        (for efficiency)'''
        nlp = spacy.load('en', disable=['parser', 'ner'])
        lemmatized_words = []
        for word in texts:
            doc = nlp(" ".join(word)) 
            lemmatized_words.append([token.lemma_ for token in doc
                            if token.pos_ in allowed_postags])
        return lemmatized_words

    # Compute Perplexity
    def perplexity(self, model, corpus):
        # a measure of how good the model is. lower the better.
        perplexity = model.log_perplexity(corpus)
        return perplexity
    
    # Compute Coherence Score
    def coherence_score(self, model, lemmatized_text, dictionary):
        coherence_model = CoherenceModel(model=model,
                                        texts=lemmatized_text,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence = coherence_model.get_coherence()
        return coherence

    def gensim_preprocessing(self, col_num, data_frame):
        self.col_num = col_num
        self.data_frame = data_frame

        self.data_frame.iloc[:,self.col_num].fillna(value='noresponse', inplace=True)
        corpus_raw = list(self.data_frame.iloc[:,self.col_num].to_numpy())
        gensim_words = list(self.gensim_sentences_to_words(corpus_raw))
        
        # Form Bigrams & Trigrams
        gensim_words_bigrams = self.make_bigrams(gensim_words)
        gensim_words_trigrams = self.make_trigrams(gensim_words_bigrams)

        # Remove Stop Words
        self.gensim_words_nostops = self.remove_stopwords(gensim_words_trigrams)

        # Do lemmatization keeping only noun, adj, vb, adv
        # self.gensim_words_lemmatized = self.lemmatization(gensim_words_nostops,
        #                          allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Create Dictionary
        self.id2word = corpora.Dictionary(self.gensim_words_nostops)

        # Create Corpus
        texts = self.gensim_words_nostops

        # Term Document Frequency
        self.gensim_corpus = [self.id2word.doc2bow(text) for text in texts]

        return self.gensim_corpus


    def gensim_standard_lda(self, num_topics=5, num_words=15):
        # Build LDA model
        gensim_lda_model = gensim.models.ldamodel.LdaModel(corpus=self.gensim_corpus,
                                                        id2word=self.id2word,
                                                        num_topics=num_topics, 
                                                        random_state=42,
                                                        passes=10,
                                                        alpha='auto',
                                                        per_word_topics=True)

        label = self.data_frame.columns.to_numpy()[self.col_num]
        print(f"Column {self.col_num} - Label: {label}\n")
        print(f"Gensim LDA Topic Modeling with {num_topics} topics:\n")
        
        # Print topics and words
        x = gensim_lda_model.show_topics(num_topics=num_topics,
                                        num_words=num_words,
                                        formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        for topic, words in topics_words:
            print(f"Topic {str(topic)}:\n{str(words)}\n")
        
        perplexity = self.perplexity(gensim_lda_model, self.gensim_corpus)
        coherence = self.coherence_score(gensim_lda_model,
                                         self.gensim_words_nostops,
                                         self.id2word)
        print(f"Perplexity: {perplexity}")
        print(f"Coherence: {coherence}")


    def gensim_mallet_lda(self, num_topics=5, num_words=15):        
        # Build LDA model
        mallet_lda_model = LdaMallet(self.mallet_path,
                        corpus=self.gensim_corpus,
                        num_topics=num_topics,
                        id2word=self.id2word)

        label = self.data_frame.columns.to_numpy()[self.col_num]
        print(f"Column {self.col_num} - Label: {label}\n")
        print(f"MALLET LDA Topic Modeling via Gensim with {num_topics} topics:\n")
        
        # Print topics and words
        x = mallet_lda_model.show_topics(num_topics=num_topics,
                                        num_words=num_words,
                                        log=False,
                                        formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        for topic, words in topics_words:
            print(f"Topic {str(topic)}:\n{str(words)}\n")
        
        coherence = self.coherence_score(mallet_lda_model,
                                         self.gensim_words_nostops,
                                         self.id2word)
        print(f"Coherence: {coherence}")
    

if __name__ == '__main__':
    nlp = SurveyNLP()
    df, df_noobs = nlp.load_survey_data()
    # 35 - Why did you start buying vinyl originally?
    # 123 - My favorite thing about Vinyl Me, Please is...
    # 124 - My LEAST favorite thing about Vinyl Me, Please is...
    nlp.gensim_preprocessing(35, df_noobs)
    nlp.gensim_mallet_lda(num_topics=5, num_words=10)
    nlp.gensim_standard_lda(num_topics=5, num_words=10)
    nlp.sklearn_lda(col_num=35,
                    data_frame=df_noobs,
                    num_topics=5,
                    num_words=10)
    nlp.sklearn_nmf(col_num=35, data_frame=df_noobs, num_words=10)
