# initial imports
import requests, logging, os, http.client, re, urllib, nltk, joblib,  spacy, math
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from googlesearch import search
from string import punctuation
from urllib.error import URLError, HTTPError
from nltk.stem.porter import PorterStemmer

nlp = spacy.load("en_core_web_lg")

# clear console
clcs = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')
clcs()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download stopwords corpora for nltk
from nltk.corpus import stopwords
try:
    stopwords.words('english')
except LookupError as e:
    nltk.download('stopwords')

# download punkt corpora for nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError as e:
    nltk.download('punkt')

MIN_WORDS = 110
MIN_UNI_WORDS = 84

# fake news classifier
classifier =  Path(__file__).parent / "fake_news_model.pkl"
fn_model = joblib.load(classifier)

# dictionary of preprocessing substitutions
subs = {
    '!': ' exclamation ',
    '?': ' question ',
    '\'': ' quotation ',
    '\"': ' quotation ',
    '”': ' quotation ',
    '“': ' quotation ',
    '’': ' apostrophe ',
}


def get_article_text(filepath):

    # Open file for reading
    try:
        FileHandler = open(filepath,"r")
        text = filepath.read_text()

    except FileNotFoundError as e:
        # file not found error.
        logging.error("File not found. Please ensure 'article.txt' is present in the file's directory.")
        return

    # Close the file
    FileHandler.close()

    # get the articles list of words, list of unique words and size of both
    word_list = text.split()
    unique = set(word_list)
    num_of_words = len(word_list)
    num_of_uni = len(unique)

    if (num_of_words > MIN_WORDS and num_of_uni > MIN_UNI_WORDS):
        # if article meets criteria
        predicted_uni = 0.93 * 3.9 * (num_of_words ** 0.67)
        if (num_of_uni >= predicted_uni):

            # break into lines and remove leading and trailing space on eac
            # break multi-headlines into a line each
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            title = text.split('\n', 1)[0] # set title to first line

            return title, text
    print(filepath, "did not meet criteria")

    return None, None


# stemming processing to be used for fake news classifier
def text_preprocess_stem(text):

    # tokenize text and substitute punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [subs.get(item,item) for item in tokens]

    # remove stopwords from text
    tokens_no_sw = [word for word in tokens if not word in stopwords.words('english')]
    # get stem of each word
    stemmer = PorterStemmer() 
    text_stemmed = [stemmer.stem(word).lower() for word in tokens_no_sw]
    processed_text = " ".join(text_stemmed)

    return processed_text


# lemmatization processing to be used for article analysis and comparison
def text_preprocess_lem(text):

    try:
        doc = nlp(text.lower())
    except Exception as e:
        err = ('Exception:', e)
        # print(err)
        return None
    
    result = []
    for token in doc:
        if token.text in stopwords.words('english'):
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    return " ".join(result)


# gathering a list of results from a google search of a query
def gather_sites(query):

    # read in blocked sites
    loc = Path(__file__).parent / "blocked sites.txt"
    with open(loc, 'r') as f:
        blocked_sites = [line.strip() for line in f]

    # blacklisted items of url, if url contains any of these, they are either incompatible or an anchor page
    blacklisted = ['.pdf','.png','jpg','#']
    sites = []

    # add first 10 found sites to a list of sites
    for i in search(query, tld = "com", num = 10, stop = 10, pause = 10, lang="en"):
        allowed = True

        # if a blacklisted substring is detected, dont add
        for substring in blacklisted:
            if substring in i:
                allowed = False
                # print('Detected unusable URL:', i)

        # if a blocked site is detected, dont add
        for blocked_site in blocked_sites:
            if blocked_site in i:
                allowed = False
                # print('Detected blocked URL:', i)

        if allowed: # if found url meets criteria, add
            sites.append(i)
    return sites


# getting text information from an article
def gather_content(url):

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resource = urllib.request.urlopen(req)
        html = resource.read().decode(resource.headers.get_content_charset('utf8'))
        
    except HTTPError as e: # exotic errors such as authentication
        err = ('Error code:', e.code)
        # print(err)
        return err

    except URLError as e: # URL error
        err = ('Reason:', e.reason)
        # print(err)
        return err

    except requests.exceptions.ConnectionError as e: # connection to page error
        err = ('Connection error:', e)
        # print(err)
        return err

    except http.client.RemoteDisconnected as disconnected_err: # timeout error, socket disconnection
        err = ('Connection aborted.', disconnected_err)
        # print(err)
        return err

    except UnicodeDecodeError as e: # decoding byte error
        err = ('Decoding error:', e.reason)
        # print(err)
        return err

    except (http.client.IncompleteRead) as e: # incomplete read error
        err = ('Incomplete read error:', e)
        # print(err)
        return err

    except Exception as e:
        err = ('Exception:', e)
        # print(err)
        return err

    # extract needed content from site
    soup = BeautifulSoup(html, "html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    list_of_lines = []

    # get header of article, most likely will be article title
    headers = soup.find('h1')
    if headers != None:
        list_of_lines.append(headers.text.strip())

    # get paragraph elements from page
    only_p = soup.find_all('p')
    for i in range(len(only_p)):
        line_text = only_p[i].text
        # this is mainly for pages that have repeating added information such as log in requests etc.
        if line_text not in list_of_lines:
            if (len(line_text.split()) > 3):
                list_of_lines.append(line_text)

    # join lines into one string
    text = '\n'.join(line for line in list_of_lines)

    # remove non latin characters
    result = re.sub(r'[^\x00-\x7f]',r'', text)
    return result


# testing purposes
def get_article_list():
    article_list = []
    e = Path(__file__).parent / "articles-read/true"
    for article in os.listdir(e):
        article_loc = e / article
        with open(article_loc) as f:
            lines = f.read()
            title = lines.split('\n', 1)[0]
        article_list.append([title, article_loc])
    return article_list

# testing purposes
def get_fake_article_list():
    article_list = []
    e = Path(__file__).parent / "articles-read/fake"
    for article in os.listdir(e):
        article_loc = e / article
        with open(article_loc) as f:
            lines = f.read()
            title = lines.split('\n', 1)[0]
        article_list.append([title, article_loc])
    return article_list


def text_sentiment(text):
    # using vader sentiment analyzer
    vader_sentiment_analyzer = SentimentIntensityAnalyzer()
    return vader_sentiment_analyzer.polarity_scores(text)


# transposes score between 0-1 to a value between 0-max
# for sentiment matching score
def map_sent_score(score, max):

    # f(x) = (-(x * root(MAX)))**2 + MAX
    # return  (-((score*(max**(1/2)))**2) + max)

    # f(x) = -MAX(x - 1)
    return -(max*(score-1))


# transposes score between 0-1 to a value between 0-max
# follows a variation of the logarithmic function, granting high results for even
# small initial scores
def map_set_score(score, max):

    # f(x) = (MAX / ln(MAX + 1)) * (ln(x*MAX + 1))
    mult = max/(math.log(max + 1))
    return  (mult * math.log((100 * score) + 1))


'''
Returns a bucket value based on compound score:
RANGE               DEGREE          SENTIMENT       VALUE
-1.00 to -0.75:     overwhelming    negative        -4
-0.75 to -0.50:     obivous         negative        -3
-0.50 to -0.25:     general         negative        -2
-0.25 to  0.00:     slight          negative        -1
 0.00         :                     neutral          0
 0.00 to  0.25:     slight          positive         1
 0.25 to  0.50:     general         positive         2
 0.25 to  0.50:     obvious         positive         3
 0.75 to  1.00:     overwhelming    positive         4
'''
def get_bucket(score):
    sentiment = ['negative',       'negative',  'negative',  'negative', 'positive', 'positive',  'positive',  'positive']
    degree =    ['overwhelmingly', 'obviously', 'generally', 'slightly', 'slightly', 'generally', 'obviously', 'overwhelmingly']
    buckets =   [-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]

    # getting sentiment value as per table

    # in case of largely neutral neutral
    if (abs(score) < 0.01):
        return 0, 'neutral', 'largely'
    else:
        for divider in range(len(buckets)):
            # check which bucket it falls under
            if (score <= buckets[divider]):
                # getting bucket based on index value
                if (divider < 4): # for negative scores
                    return (divider - 4), sentiment[divider], degree[divider]
                else: # positive scores
                    return (divider - 3), sentiment[divider], degree[divider]



# article_list = get_article_list()

# e = Path(__file__).parent / "keywords.txt"

# f = open(e, "w")

# for article in article_list:
#     switch = get_article_text(article)
#     f.write(str(article))
#     f.write('\n')
#     if switch == None:
#         f.write('Article does not contain enough words for this program.')
#         f.write('\n\n')
#         # print(f'{article} does not contain enough words for this program.\n')
#         continue
#     # print(summarize(switch, 0.05))
#     f.writelines(summarize(switch, 0.05))
#     f.write('\n\n')

# f.close()
