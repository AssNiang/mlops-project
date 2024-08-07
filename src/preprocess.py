"""Data processing Module"""

import re
import string
from loguru import logger
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import spacy


# Dropping missing values and duplicated observations
def drop_missing_and_duplicates(data):
    """
    Function to preprocess the data by logging the count of missing values and duplicates,
    dropping missing and duplicate observations, resetting the index, and encoding labels.

    Args:
    data (pd.DataFrame): Input dataframe with a 'label' column to encode.

    Returns:
    pd.DataFrame: Preprocessed dataframe.
    """
    # Calculate missing values and duplicate observations
    missing_values_count = len(data) - len(data.dropna())
    duplicate_observations_count = data.duplicated().sum()

    # Log number of observations with missing values
    logger.info(f"Number of observations with missing values: {missing_values_count}")

    # Log number of duplicate observations
    logger.info(f"Number of duplicate observations: {duplicate_observations_count}")

    # Dropping observations with missing values
    logger.info("Dropping observations with missing values")
    data.dropna(inplace=True)  

    # Dropping duplicate observations
    logger.info("Dropping duplicate observations")
    data.drop_duplicates(inplace=True)  

    # Resetting index
    logger.info("Resetting index")
    data.reset_index(drop=True, inplace=True)  

    # Manual encoding of labels
    logger.info("Encoding of labels")
    label_dict = {"Electronics": 0, "Household": 1, "Books": 2, "Clothing & Accessories": 3}
    data.replace({"label": label_dict}, inplace=True)

    # Log dataset shape
    logger.info(f"Dataset shape: {data.shape}")

    return data


# RegexpTokenizer
regexp = RegexpTokenizer(r"[\w']+")


# Converting to lowercase
def convert_to_lowercase(text):
    """Converts a string to lowercase

    Args:
        text (str): text to convert

    Returns:
        text: text in lowercase
    """
    return text.lower()


# Removing whitespaces
def remove_whitespace(text):
    """Removes whitespace from a string

    Args:
        text (str): text to process

    Returns:
        text: text with whitespace removed
    """
    return text.strip()


# Removing punctuations
def remove_punctuation(text):
    """Removes punctuation from a string

    Args:
        text (str): text to process

    Returns:
        text: text with punctuation removed
    """
    punct_str = string.punctuation
    punct_str = punct_str.replace("'", "")  # discarding apostrophe from the string to keep the contractions intact
    return text.translate(str.maketrans("", "", punct_str))


# Removing HTML tags
def remove_html(text):
    """Removes html from a string

    Args:
        text (str): text to process

    Returns:
        text: text with html removed
    """
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


# Removing emojis
def remove_emoji(text):
    """Removes emoji from a string

    Args:
        text (str): text to process

    Returns:
        text: text with emoji removed
    """
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


# Removing other unicode characters
def remove_http(text):
    """Removes URLs starting with http or https from a string.

    Args:
        text (str): Text to process.

    Returns:
        str: Text with URLs removed.
    """
    # Regular expression to match URLs starting with http, https, or www.
    http_pattern = r"https?://\S+|www\.\S+"

    # Using re.sub to replace the matched URLs with an empty string.
    cleaned_text = re.sub(http_pattern, "", text)
    return cleaned_text


# List of acronyms
ACRONYMS_URL = (
    "https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/JSON/english_acronyms.json"
)
acronyms_dict = pd.read_json(ACRONYMS_URL, typ="series")
acronyms_list = list(acronyms_dict.keys())


# Function to convert acronyms in a text
def convert_acronyms(text):
    """Converts acronyms into their full representation

    Args:
        text (str): text to convert

    Returns:
        text: text with acronyms converted
    """
    words = []
    for word in regexp.tokenize(text):
        if word in acronyms_list:
            words = words + acronyms_dict[word].split()
        else:
            words = words + word.split()

    text_converted = " ".join(words)
    return text_converted


# List of contractions
CONTRACTIONS_URL = (
    "https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/JSON/english_contractions.json"
)
contractions_dict = pd.read_json(CONTRACTIONS_URL, typ="series")
contractions_list = list(contractions_dict.keys())


# Function to convert contractions in a text
def convert_contractions(text):
    """Converts contractions into their full representation

    Args:
        text (str): text to convert

    Returns:
        text: text with contractions converted
    """
    words = []
    for word in regexp.tokenize(text):
        if word in contractions_list:
            words = words + contractions_dict[word].split()
        else:
            words = words + word.split()

    text_converted = " ".join(words)
    return text_converted


# Stopwords
stops = stopwords.words("english")  # stopwords
addstops = [
    "among",
    "onto",
    "shall",
    "thrice",
    "thus",
    "twice",
    "unto",
    "us",
    "would",
]  # additional stopwords
allstops = stops + addstops


# Function to remove stopwords from a list of texts
def remove_stopwords(text):
    """Removes stopwords from the text

    Args:
        text (str): text to process

    Returns:additional
        text: text with  stopwords removed
    """
    return " ".join([word for word in regexp.tokenize(text) if word not in allstops])


# pyspellchecker
spell = SpellChecker()


def pyspellchecker(text):
    """Checks if the text contains errors or unknown words."
    Args:
        text (str): text to correct

    Returns:
        text: text corrected
    """
    word_list = regexp.tokenize(text)
    word_list_corrected = []
    for word in word_list:
        if word in spell.unknown(word_list):
            word_corrected = spell.correction(word)
            if word_corrected is None:
                word_list_corrected.append(word)
            else:
                word_list_corrected.append(word_corrected)
        else:
            word_list_corrected.append(word)
    text_corrected = " ".join(word_list_corrected)
    return text_corrected


# Lemmatization
spacy_lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def text_lemmatizer(text):
    """Keep only the root of the words

    Args:
        text (str): text to process

    Returns:
        text: text processed
    """
    text_spacy = " ".join([token.lemma_ for token in spacy_lemmatizer(text)])
    return text_spacy


# Discardment of non-alphabetic words
def discard_non_alpha(text):
    """Discards non-alphabetic words from the text

    Args:
        text (str): text to process

    Returns:
        text processed
    """
    word_list_non_alpha = [word for word in regexp.tokenize(text) if word.isalpha()]
    text_non_alpha = " ".join(word_list_non_alpha)
    return text_non_alpha


def keep_pos(text):
    """Keeps Part-Of-Speech tags like nouns, proper nouns, adjectives, adverbs, verbs, pronouns, possessive pronouns, etc. from the text.

    Args:
        text (str): text to process

    Returns:
    text: text with only Part-Of-Speech tags kept like nouns, proper nouns, adjectives, adverbs, verbs, pronouns, possessive pronouns, etc.
    """
    tokens = regexp.tokenize(text)
    tokens_tagged = nltk.pos_tag(tokens)
    # keep_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'FW']
    keep_tags = [
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "FW",
        "PRP",
        "PRPS",
        "RB",
        "RBR",
        "RBS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WPS",
        "WRB",
    ]
    keep_words = [x[0] for x in tokens_tagged if x[1] in keep_tags]
    return " ".join(keep_words)


# Additional stopwords
alphabets = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
prepositions = [
    "about",
    "above",
    "across",
    "after",
    "against",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beside",
    "between",
    "by",
    "down",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "off",
    "on",
    "out",
    "over",
    "through",
    "to",
    "toward",
    "under",
    "up",
    "with",
]
prepositions_less_common = [
    "aboard",
    "along",
    "amid",
    "as",
    "beneath",
    "beyond",
    "but",
    "concerning",
    "considering",
    "despite",
    "except",
    "following",
    "like",
    "minus",
    "onto",
    "outside",
    "per",
    "plus",
    "regarding",
    "round",
    "since",
    "than",
    "till",
    "underneath",
    "unlike",
    "until",
    "upon",
    "versus",
    "via",
    "within",
    "without",
]
coordinating_conjunctions = ["and", "but", "for", "nor", "or", "so", "and", "yet"]
correlative_conjunctions = [
    "both",
    "and",
    "either",
    "or",
    "neither",
    "nor",
    "not",
    "only",
    "but",
    "whether",
    "or",
]
subordinating_conjunctions = [
    "after",
    "although",
    "as",
    "as if",
    "as long as",
    "as much as",
    "as soon as",
    "as though",
    "because",
    "before",
    "by the time",
    "even if",
    "even though",
    "if",
    "in order that",
    "in case",
    "in the event that",
    "lest",
    "now that",
    "once",
    "only",
    "only if",
    "provided that",
    "since",
    "so",
    "supposing",
    "that",
    "than",
    "though",
    "till",
    "unless",
    "until",
    "when",
    "whenever",
    "where",
    "whereas",
    "wherever",
    "whether or not",
    "while",
]
others = [
    "ã",
    "å",
    "ì",
    "û",
    "ûªm",
    "ûó",
    "ûò",
    "ìñ",
    "ûªre",
    "ûªve",
    "ûª",
    "ûªs",
    "ûówe",
]
additional_stops = (
    alphabets
    + prepositions
    + prepositions_less_common
    + coordinating_conjunctions
    + correlative_conjunctions
    + subordinating_conjunctions
    + others
)


def remove_additional_stopwords(text):
    """Removes additional stopwords from the text

    Args:
        text (str): text to process

    Returns:
        text: text with additional stopwords removed
    """
    return " ".join([word for word in regexp.tokenize(text) if word not in additional_stops])


def text_normalizer(text):
    """Normalizes the text based on all the other functions

    Args:
        text (str): text to process

    Returns:
        text: normalized text
    """
    text = convert_to_lowercase(text)
    text = remove_whitespace(text)
    text = re.sub(r"\n", "", text)  # converting text to one line
    text = re.sub(r"\[.*?\]", "", text)  # removing square brackets
    text = remove_http(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = convert_acronyms(text)
    text = convert_contractions(text)
    text = remove_stopwords(text)
    #     text = pyspellchecker(text)
    text = text_lemmatizer(text)  # text = text_stemmer(text)
    text = discard_non_alpha(text)
    text = keep_pos(text)
    text = remove_additional_stopwords(text)
    return text
