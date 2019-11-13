import os
import sys
sys.path.append('.')

import pandas as pd

# SKLearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Import survey data loading script
from src.data.make_survey_dataset import load_data_as_dataframe

### ----- Set up project directory path names to load and save data ----- ###
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')


class SurveyNLP(object):
    """Queries PostgreSQL database using psycopg2 and performs NLP.

    Reads in raw text data for one open-ended Vinyl Me, Please customer survey
    question and performs various types of NLP. Depending on options set and
    methods called, the following options are available:
        - SKLearn LDA
        - SKLearn NMF
        - Gensim LDA
        - Mallet LDA using Gensim wrapper.
    For more info on Mallet LDA visit the homepage: http://mallet.cs.umass.edu

    Returns:
        Top n topics with top n words for each topic.
    """

    def __init__(self):
        # Adding to SKLearn stop words list
        self.stop_words = text.ENGLISH_STOP_WORDS.union(['vmp',
                                                        'really',
                                                        'like',
                                                        'na',
                                                        'noresponse'])
        # Path to Mallet LDA install directory.
        self.mallet_path = os.path.join(ROOT_DIRECTORY,
                                        "../mallet-2.0.8/bin/mallet")

        # Using imported script to load data from safe directory.
        self.df, self.df_col_names = load_data_as_dataframe(
                                  filename='2019 Member Survey - Raw Data.csv')
        print(f"Loaded survey DataFrame of size {self.df.shape}.\n")

    def create_df_noobs(self, source_df):
        """Creates additional DataFrame for vinyl noobs.

        Vinyl noobs are defined as people who have been buying vinyl for 0-3
        years. This subset of customers can be useful to look at.

        Returns:
            df_noobs - New DataFrame containing noobs data.
        """

        self.df_noobs = source_df[(source_df.iloc[:,33] == 'I just started') 
                         | (source_df.iloc[:,33] == '6 - 12 months') 
                         | (source_df.iloc[:,33] == '1-3 years')]
        return self.df_noobs

    def display_sklearn_topics(self, model, feature_names, num_words):
        """Prints sorted feature names and topics for SKLearn models.

        Args:
            model: Trained SKLearn NLP model.
            feature_names (list of int): List of feature names from model.
            num_words (int): Number of words to print for each topic.
        """

        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx}:")
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-num_words - 1:-1]]))

    def sklearn_lda(self, data_frame, col_num, num_topics=5, num_words=15):
        """Performs LDA on given column.

        Args:
            data_frame: DataFrame in question.
            col_num (int): Column index to analyze via LDA.
            num_topics (int): Desired number of topics to model.
            num_words (int): Number of words to print for each topic.
        """

        # Filling NaN values with 'noresponse', which was added to stopwords.
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

    def sklearn_nmf(self, data_frame, col_num, num_topics=5, num_words=15):
        """Performs NMF on given column.

        Args:
            data_frame: DataFrame in question.
            col_num (int): Column index to analyze via NMF.
            num_topics (int): Desired number of topics to model.
            num_words (int): Number of words to print for each topic.
        """

        # Filling NaN values with 'noresponse', which was added to stopwords.
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

        sklearn_nmf = NMF(n_components=num_topics,
                          init='random',
                          random_state=42)

        W = sklearn_nmf.fit_transform(sklearn_trans_text)
        H = pd.DataFrame(sklearn_nmf.components_)
        label = data_frame.columns.to_numpy()[col_num]
        print(f"Column {col_num} - Label: {label}\n")
        print(f"SKLearn NMF Topic Modeling with {num_topics} topics:\n")
        self.display_sklearn_topics(sklearn_nmf,
                                    sklearn_feature_names,
                                    num_words)

    def gensim_sentences_to_words(self, sentences):
        """Converts each response into a list of tokens.

        For each response, gensim.utils.simple_preprocess function lowercases
        and tokenizes the string.
        """
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence)))

    def make_bigrams(self, texts):
        """Makes bigrams. Higher threshold = fewer phrases."""

        self.bigram = gensim.models.Phrases(texts, min_count=5, threshold=20)
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_model = gensim.models.phrases.Phraser(self.bigram)
        return [self.bigram_model[doc] for doc in texts]

    def make_trigrams(self, texts):
        """Makes trigrams. Higher threshold = fewer phrases."""
        
        self.trigram = gensim.models.Phrases(self.bigram[texts], threshold=15)      
        # Faster way to get a sentence clubbed as a trigram/bigram
        self.trigram_model = gensim.models.phrases.Phraser(self.trigram)
        return [self.trigram_model[self.bigram_model[doc]] for doc in texts]

    def remove_stopwords(self, texts):
        """Removes stop words using previously definded SKLearn list."""

        return [[word for word in simple_preprocess(str(doc))
                if word not in self.stop_words] for doc in texts]

    def lemmatization(self,
                      texts,
                      allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Initialize spacy lemmatizer, keeping only tagger (for efficiency).

        See https://spacy.io/api/annotation for more info.

        Returns:
            lemmatized_words = List of lemmatized word strings.
        """

        nlp = spacy.load('en', disable=['parser', 'ner'])
        lemmatized_words = []
        for word in texts:
            doc = nlp(" ".join(word)) 
            lemmatized_words.append([token.lemma_ for token in doc
                            if token.pos_ in allowed_postags])
        return lemmatized_words

    def perplexity(self, model, corpus):
        """Computes complexity to score model. lower = better."""

        perplexity = model.log_perplexity(corpus)
        return perplexity
    
    def coherence_score(self, model, lemmatized_text, dictionary):
        """Calculates topic coherence for Gensim models."""

        coherence_model = CoherenceModel(model=model,
                                         texts=lemmatized_text,
                                         dictionary=dictionary,
                                         coherence='c_v')
        coherence = coherence_model.get_coherence()
        return coherence

    def gensim_preprocessing(self, col_num, data_frame):
        """Performs preprocessing required before running Gensim algorithms."""

        self.col_num = col_num
        self.data_frame = data_frame

        # Filling NaN values with 'noresponse', which was added to stopwords.
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
        """Performs Gensim standard/built-in LDA.

        Requires gensim_corpus output for a column from gensim_preprocessing().

        Args:
            num_topics (int): Desired number of topics to model.
            num_words (int): Number of words to print for each topic.
        """

        gensim_lda_model = gensim.models.ldamodel.LdaModel(
                                                  corpus=self.gensim_corpus,
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
        """Performs Mallet LDA using Gensim wrapper.

        Requires gensim_corpus output for a column from gensim_preprocessing().

        Args:
            num_topics (int): Desired number of topics to model.
            num_words (int): Number of words to print for each topic.
        """

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
    df_noobs = nlp.create_df_noobs(nlp.df)
    # 35 - Why did you start buying vinyl originally?
    # 123 - My favorite thing about Vinyl Me, Please is...
    # 124 - My LEAST favorite thing about Vinyl Me, Please is...
    column_idx = 124
    num_words = 15

    nlp.gensim_preprocessing(column_idx,
                             df_noobs)
    nlp.gensim_mallet_lda(num_topics=5,
                          num_words=num_words)
    nlp.gensim_standard_lda(num_topics=5,
                            num_words=num_words)
    nlp.sklearn_lda(data_frame=df_noobs,
                    col_num=column_idx,
                    num_topics=5,
                    num_words=num_words)
    nlp.sklearn_nmf(data_frame=df_noobs,
                    col_num=column_idx,
                    num_words=num_words)
