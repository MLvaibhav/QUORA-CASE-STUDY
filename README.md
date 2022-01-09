# QUORA-CASE-STUDY

Business Problem 
 
 Description
 
 Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.
 
 Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
 
 
 __ Problem Statement __
 
Identify which questions asked on Quora are duplicates of questions that have already been asked.

This could be useful to instantly provide answers to questions that have already been answered.

We are tasked with predicting whether a pair of questions are duplicates or not.

Type of Machine Leaning Problem 

It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

Exploratory Data Analysis

Importing libraries :

```python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
#import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
```

Reading data and basic stats

```python
df = pd.read_csv("train.csv")

print("Number of data points:",df.shape[0])
```

```python
df.head()

	id	qid1	qid2	question1	                               question2	                                     is_duplicate
0	0	1	2	What is the step by step guide to invest in sh...	What is the step by step guide to invest in sh...	0
1	1	3	4	What is the story of Kohinoor (Koh-i-Noor) Dia...	What would happen if the Indian government sto...	0
2	2	5	6	How can I increase the speed of my internet co...	How can Internet speed be increased by hacking...	0
3	3	7	8	Why am I mentally very lonely? How can I solve...	Find the remainder when [math]23^{24}[/math] i...	0
4	4	9	10	Which one dissolve in water quikly sugar, salt...	Which fish would survive in salt water?	0

```

We are given a minimal number of data fields here, consisting of:
id: Looks like a simple rowID.

qid{1, 2}: The unique ID of each question in the pair.

question{1, 2}: The actual textual contents of the questions.

is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

```python
df.groupby("is_duplicate")['id'].count().plot.bar()
```

![Screen Shot 2021-10-15 at 12 45 58 PM](https://user-images.githubusercontent.com/90976062/137447423-ef7dfdaf-007a-4030-ae78-6b7f53ebc95f.png)

```python
print('~> Total number of question pairs for training:\n   {}'.format(len(df)))

~> Total number of question pairs for training:
   404290
```

below we are getting percentage of similar and not similar question pairs , in code we are picking is_duplicate column.

taking its mean and multiplying it with 100 to get percentage and rounding output upto 2 decimal places .

```python
print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))

~> Question pairs are not Similar (is_duplicate = 0):
   63.08%

~> Question pairs are Similar (is_duplicate = 1):
   36.92%
 ```
 
 Below we are finding no of unique questions :
 
 ```python
 #here we are converting each qid's to list and storing all together in form of series (
#Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc. )

qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unique_qs = len(np.unique(qids))
#qids.value_counts() will count occurence of each of value in series qids and sort them in descending order of occurence 
qs_morethan_onetime = np.sum(qids.value_counts() > 1)
print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
#print len(np.unique(qids))

print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))

print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 

q_vals=qids.value_counts()

q_vals=q_vals.values

Total num of  Unique Questions are: 537933

Number of unique questions that appear more than one time: 111780 (20.77953945937505%)

Max number of times a single question is repeated: 157
```

Plotting above outcome 

```python
x = ["unique_questions" , "Repeated Questions"]
y =  [unique_qs , qs_morethan_onetime]

plt.figure(figsize=(10, 6))
plt.title ("Plot representing unique and repeated questions  ")
sns.barplot(x,y)
plt.show()
```

![Screen Shot 2021-10-15 at 1 16 04 PM](https://user-images.githubusercontent.com/90976062/137451311-c72b8b6c-917a-4a06-9e4a-c92e72a23761.png)

Checking for Duplicates

```python
#checking whether there are any repeated pair of questions

pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate questions",(pair_duplicates).shape[0] - df.shape[0])
```

Checking for NULL values

```python
#Checking whether there are any rows with null values
nan_rows = df[df.isnull().any(1)] # any(1) means look row wise instead of column wise 
print (nan_rows)
```

```python
# Filling the null values with ' '
df = df.fillna('')
nan_rows = df[df.isnull().any(1)]
print (nan_rows)

Empty DataFrame
Columns: [id, qid1, qid2, question1, question2, is_duplicate]
Index: []
```

Basic Feature Extraction (before cleaning) 

Let us now construct a few features like:

freq_qid1 = Frequency of qid1's

freq_qid2 = Frequency of qid2's

q1len = Length of q1

q2len = Length of q2

q1_n_words = Number of words in Question 1

q2_n_words = Number of words in Question 2

word_Common = (Number of common unique words in Question 1 and Question 2)

word_Total =(Total num of words in Question 1 + Total num of words in Question 2)

word_share = (word_common)/(word_Total)

freq_q1+freq_q2 = sum total of frequency of qid1 and qid2

freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

```python
#os.path.isfile checks file mentioned is present or not if present read tht csv file or else create all the above features as new columns and convert it into csv file to be stored as trained dataset
if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
else:
    # groups it on basis of firstly qid1 and then qid2 then count no of rows present in that particular group which is the frequency
    
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len() 
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
#below function will get yu common words in both questions firstly w1 and w2 are set of words in lowercase 
# and we return intersection of both sets of words that will result in common words between both.

#.strip() method removes any leading (spaces at the beginning) and trailing (spaces at the end) characters (space is the default leading character to remove)

    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)
#here we using lambda function on data that is present after comma and finally storing output in w1 
    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    df['word_Total'] = df.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    df.to_csv("df_fe_without_preprocessing_train.csv",index=False)

df.head(10)
```
```python
Analysis of some of the extracted features 

Here are some questions have only one single words.


print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))

print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))

print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])
print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])

Minimum length of the questions in question1 :  1
Minimum length of the questions in question2 :  1
Number of Questions with minimum length [question1] : 67
Number of Questions with minimum length [question2] : 24
```

Analysing feature - word_share

```python
#So from below pdf of is_duplicate =0 and 1 we can say where word_share is higher there are higher change of duplicacy.
# though both pdf are overlapping but not completely overlapping so this feature can be useful 
plt.figure(figsize=(12, 8))
#below statement of subplot means figure should have one row and two columns like drawn below both plots are drawn in a 
#single row and side by side as two columns voilenplot is the first thats why (1,2,1)
plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )
plt.show()
```

![Screen Shot 2021-10-17 at 10 20 19 PM](https://user-images.githubusercontent.com/90976062/137637070-387b392b-3eeb-4183-95a5-fd54090b4ccf.png)

The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

Analyse feature : word_common

```python

plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )
plt.show()
```
![Screen Shot 2021-10-17 at 10 24 29 PM](https://user-images.githubusercontent.com/90976062/137637195-4f9a5437-f57a-4515-9e26-6d7767a90c8b.png)
The distributions of the word_Common feature in similar and non-similar questions are highly overlapping

Preprocessing of Text & Advanced feature selection 

Preprocessing:

Removing html tags

Removing Punctuations

Performing stemming

Removing Stopwords

Expanding contractions etc.

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
#import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# This package is used for finding longest common subsequence between two strings
# you can write your own dp code for this
#import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
# Import the Required lib packages for WORD-Cloud generation
# https://stackoverflow.com/questions/45625434/how-to-install-wordcloud-in-python3-6
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image
```

```python
# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")

# 000000 replaced by millions m similarly 000 replaced by thousands k
def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
```

Advanced Feature Extraction (NLP and Fuzzy Features)

Definition:

Token: You get a token by splitting sentence a space

Stop_Word : stop words as per NLTK.

Word : A token that is not a stop_word

Features:

cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2 

cwc_min = common_word_count / (min(len(q1_words), len(q2_words))


cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2 

cwc_max = common_word_count / (max(len(q1_words), len(q2_words))


csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 

csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))


csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2

csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))


ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2

ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))


ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2

ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))


last_word_eq : Check if last word of both questions is equal or not

last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])


first_word_eq : Check if First word of both questions is equal or not

first_word_eq = int(q1_tokens[0] == q2_tokens[0])


abs_len_diff : Abs. length difference

abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))


mean_len : Average Token Length of both Questions

mean_len = (len(q1_tokens) + len(q2_tokens))/2

#below mwntion fuzz features are different ways to measure similarity bwetween two given sentences #fuzz_ratio feature works like this it basically calculates no of edits to be

done if we want to make both sentences to similar lesser no of edits higher fuzz_ratio eg sentence 1 - new york mets and sentence 2 - new york meats if we want to make sentence 1 similar to sentence 2 we just have to make only one edit add a in mets so fuzz_ratio for these two will be very high.

fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

so in fuzz_partial_ratio it checks is there a partial match between two senetence eg sentence 1 yankees and sentence 2 new york yankees in both yankees is same so its a perfet

partial match giving fuzz_partial_ratio to be 100 on the other hand if we had used fuzz_ratio that would be have given lower value as we need to do more edits to make sentence 

1 similar to sentence 2 but if we see both sentence both are basically same so its better to use partial ratio in this case

fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

here we divide sentence in tokens, sort them and then get their fuzz_ratio

token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

here we also divide sentence into token , sort them after that forstly we we take sorted intersection of both sorted tokens lets say its t0 , then we take sorted intersection between both + left part(rest of word) in first tokenised sentence let say it came aout as t1 then we again take sorted intersection + left part form second tokenised sentence then finally we take fuzz_ratio of all combinations that is fuzz_ratio of t0 ,t1 fuzz_ratio of t1 ,t2 fuzz_ratio of t0 ,t2 and finally pick value having highest fuzz_ratio
token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2

longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

```python
def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    # preprocessing each question
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    
    # Merging Features with dataset
    
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
   
    #Computing Fuzzy Features and Merging with Dataset
    
    # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy
    print("fuzzy features..")

    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df
    ```
   Below code first searches if the mentioned file is present if not then it will use train.csv apply all preprocessing done above and finally store new file in csv format. 
    ```python
    if os.path.isfile('nlp_features_train.csv'):
    df = pd.read_csv("nlp_features_train.csv",encoding='latin-1')
    df.fillna('')
else:
    print("Extracting features for train:")
    df = pd.read_csv("train.csv")
    df = extract_features(df)
    df.to_csv("nlp_features_train.csv", index=False)
df.head(2)
```

Analysis of extracted features

Plotting Word clouds

Creating Word Cloud of Duplicates and Non-Duplicates Question pairs

We can observe the most frequent occuring words

```python
df_duplicate = df[df['is_duplicate'] == 1]
dfp_nonduplicate = df[df['is_duplicate'] == 0]

# Converting 2d array of q1 and q2 and flatten the array: like {{1,2},{3,4}} to {1,2,3,4}
p = np.dstack([df_duplicate["question1"], df_duplicate["question2"]]).flatten()
n = np.dstack([dfp_nonduplicate["question1"], dfp_nonduplicate["question2"]]).flatten()

print ("Number of data points in class 1 (duplicate pairs) :",len(p))
print ("Number of data points in class 0 (non duplicate pairs) :",len(n))

#Saving the np array into a text file
np.savetxt('train_p.txt', p, delimiter=' ', fmt='%s')
np.savetxt('train_n.txt', n, delimiter=' ', fmt='%s')

Number of data points in class 1 (duplicate pairs) : 298526

Number of data points in class 0 (non duplicate pairs) : 510054
```


# reading the text files and removing the Stop Words:
d = path.dirname('.')
print(d)

textp_w = open(path.join(d, 'train_p.txt')).read()
textn_w = open(path.join(d, 'train_n.txt')).read()
#print(d)

```python
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("br")
stopwords.add(" ")
stopwords.remove("not")

stopwords.remove("no")
#stopwords.remove("good")
#stopwords.remove("love")
stopwords.remove("like")
#stopwords.remove("best")
#stopwords.remove("!")
print ("Total number of words in duplicate pair questions :",len(textp_w))
print ("Total number of words in non duplicate pair questions :",len(textn_w))
```

```python
wc = WordCloud(background_color="white", max_words=len(textp_w), stopwords=stopwords)
wc.generate(textp_w)
print ("Word Cloud for Duplicate Question pairs")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![Screen Shot 2021-10-18 at 9 11 52 AM](https://user-images.githubusercontent.com/90976062/137666005-4d82ccd1-6069-41ea-ab24-d2e5da52f11f.png)

```python
wc = WordCloud(background_color="white", max_words=len(textn_w),stopwords=stopwords)
# generate word cloud
wc.generate(textn_w)
print ("Word Cloud for non-Duplicate Question pairs:")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![Screen Shot 2021-10-18 at 9 14 09 AM](https://user-images.githubusercontent.com/90976062/137666169-54e43504-75d4-4129-a593-21a9e6d1c584.png)

We are now picking some of our features that we created to plot pair plots that is bivariate analysis

```python
n = df.shape[0]
sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:n], hue='is_duplicate', vars=['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'])
plt.show()
```

![Screen Shot 2021-10-18 at 9 40 57 AM](https://user-images.githubusercontent.com/90976062/137668128-4b06cdbe-837d-4f8b-b903-234b0845c26b.png)

From above pair plots few take aways are :

1. there is some kind of separation between duplicate and non duplicate words for almost all features that makes features useful.

2. if we see barplot for ctc_min we see as ctc_min increases word duplicacy also increased and for tken_sort_ratio with increase in tht value there are more duplicate words 

lets do univariate analsis 

```python
# Distribution of the token_sort_ratio
plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'token_sort_ratio', data = df[0:] , )

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['token_sort_ratio'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['token_sort_ratio'][0:] , label = "0" , color = 'blue' )
plt.show()
```
![Screen Shot 2021-10-18 at 9 57 06 AM](https://user-images.githubusercontent.com/90976062/137669335-05e939d6-a8a7-42f2-acb8-3141a8f6bb11.png)

From above plot we can see as the token_sort_ratio increases duplcacy increases , there is a overlap but still this feature is important .

As we have 15 features if we can visualize all of them to see outcome that will be great here comes the use of Data Visualization using TSNE.

What TSNE does is it embeds 15 dimentional to smaller dimension while preserving distance between only neighborhood points.

we will reduce 15 dimentional data into 2 dimension and then visualse it 

```python
# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

from sklearn.preprocessing import MinMaxScaler

dfp_subsampled = df[0:5000]
X = MinMaxScaler().fit_transform(dfp_subsampled[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
y = dfp_subsampled['is_duplicate'].values
```
```python
tsne2d = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2,
    angle=0.5
).fit_transform(X)
```

```python
df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(30, 1000))
plt.show()
```
![Screen Shot 2021-10-18 at 10 27 53 AM](https://user-images.githubusercontent.com/90976062/137671645-57138d3b-91e1-495d-a0b4-6d76083d473c.png)

perplexity is no of neighbor whose distance want to preserve 

we  can see from plot with only one run we can see red points are fairly distinguishable at many spots in plot making our 15 feature useful .

we can run it multiple times with different value of no steps and perplexity to get better plot 

Now lets apply TFIDF vectorizer on text features to convert them into vectors and normalizer on numeric features 

```python
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
```

```python
df = pd.read_csv("nlp_features_train.csv")

#converting questions in unicode  format as python handles strings in that format
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))
```

Droping is_duplicate feature from dataset and placing it on other variable 


```python
y = df['is_duplicate'].values
x = df.drop(['is_duplicate'],axis = 1)
```
train test split 

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
```
Applying vectorizer on both text features question 1 and question 2

```python
from sklearn.feature_extraction.text import TfidfVectorizer

#questions1 = list(df['question1'].values)
#questions2 = list(df['question2'].values)
tfidf = TfidfVectorizer(lowercase=False )
tfidf.fit(X_train['question1'].values)
quest1_train = tfidf.transform(X_train['question1'].values) 
quest1_cv = tfidf.transform(X_cv['question1'].values)
quest1_test = tfidf.transform(X_test['question1'].values)

tfidf2 = TfidfVectorizer(lowercase=False )
tfidf2.fit(X_train['question2'].values)
quest2_train = tfidf2.transform(X_train['question2'].values)
quest2_cv = tfidf2.transform(X_cv['question2'].values)
quest2_test = tfidf2.transform(X_test['question2'].values)
```
Normalizing numerical features 

```python
from sklearn.preprocessing import Normalizer
normalizer1 = Normalizer()
normalizer1.fit(X_train['cwc_min'].values.reshape(-1,1))
cwc_train = normalizer1.transform(X_train['cwc_min'].values.reshape(-1,1))
cwc_cv = normalizer1.transform(X_cv['cwc_min'].values.reshape(-1,1))
cwc_test = normalizer1.transform(X_test['cwc_min'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer2 = Normalizer()
normalizer2.fit(X_train['cwc_max'].values.reshape(-1,1))
cwcmax_train = normalizer2.transform(X_train['cwc_max'].values.reshape(-1,1))
cwcmax_cv = normalizer2.transform(X_cv['cwc_max'].values.reshape(-1,1))
cwcmax_test = normalizer2.transform(X_test['cwc_max'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer3 = Normalizer()
normalizer3.fit(X_train['csc_min'].values.reshape(-1,1))
csc_min_train = normalizer3.transform(X_train['csc_min'].values.reshape(-1,1))
csc_min_cv = normalizer3.transform(X_cv['csc_min'].values.reshape(-1,1))
csc_min_test = normalizer3.transform(X_test['csc_min'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer4 = Normalizer()
normalizer4.fit(X_train['csc_max'].values.reshape(-1,1))
csc_max_train = normalizer4.transform(X_train['csc_max'].values.reshape(-1,1))
csc_max_cv = normalizer4.transform(X_cv['csc_max'].values.reshape(-1,1))
csc_max_test = normalizer4.transform(X_test['csc_max'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer5 = Normalizer()
normalizer5.fit(X_train['ctc_min'].values.reshape(-1,1))
ctc_min_train = normalizer5.transform(X_train['ctc_min'].values.reshape(-1,1))
ctc_min_cv = normalizer5.transform(X_cv['ctc_min'].values.reshape(-1,1))
ctc_min_test = normalizer5.transform(X_test['ctc_min'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer6 = Normalizer()
normalizer6.fit(X_train['ctc_max'].values.reshape(-1,1))
ctc_max_train = normalizer6.transform(X_train['ctc_max'].values.reshape(-1,1))
ctc_max_cv = normalizer6.transform(X_cv['ctc_max'].values.reshape(-1,1))
ctc_max_test = normalizer6.transform(X_test['ctc_max'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer7 = Normalizer()
normalizer7.fit(X_train['last_word_eq'].values.reshape(-1,1))
last_word_eq_train = normalizer7.transform(X_train['last_word_eq'].values.reshape(-1,1))
last_word_eq_cv = normalizer7.transform(X_cv['last_word_eq'].values.reshape(-1,1))
last_word_eq_test = normalizer7.transform(X_test['last_word_eq'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer8 = Normalizer()
normalizer8.fit(X_train['first_word_eq'].values.reshape(-1,1))
first_word_eq_train = normalizer8.transform(X_train['first_word_eq'].values.reshape(-1,1))
first_word_eq_cv = normalizer8.transform(X_cv['first_word_eq'].values.reshape(-1,1))
first_word_eq_test = normalizer8.transform(X_test['first_word_eq'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer9 = Normalizer()
normalizer9.fit(X_train['abs_len_diff'].values.reshape(-1,1))
abs_len_diff_train = normalizer9.transform(X_train['abs_len_diff'].values.reshape(-1,1))
abs_len_diff_cv = normalizer9.transform(X_cv['abs_len_diff'].values.reshape(-1,1))
abs_len_diff_test = normalizer9.transform(X_test['abs_len_diff'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer10 = Normalizer()
normalizer10.fit(X_train['mean_len'].values.reshape(-1,1))
mean_len_train = normalizer10.transform(X_train['mean_len'].values.reshape(-1,1))
mean_len_cv = normalizer10.transform(X_cv['mean_len'].values.reshape(-1,1))
mean_len_test = normalizer10.transform(X_test['mean_len'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer11 = Normalizer()
normalizer11.fit(X_train['token_set_ratio'].values.reshape(-1,1))
token_set_ratio_train = normalizer11.transform(X_train['token_set_ratio'].values.reshape(-1,1))
token_set_ratio_cv = normalizer11.transform(X_cv['token_set_ratio'].values.reshape(-1,1))
token_set_ratio_test = normalizer11.transform(X_test['token_set_ratio'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer12 = Normalizer()
normalizer12.fit(X_train['token_sort_ratio'].values.reshape(-1,1))
token_sort_ratio_train = normalizer12.transform(X_train['token_sort_ratio'].values.reshape(-1,1))
token_sort_ratio_cv = normalizer12.transform(X_cv['token_sort_ratio'].values.reshape(-1,1))
token_sort_ratio_test = normalizer12.transform(X_test['token_sort_ratio'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer13 = Normalizer()
normalizer13.fit(X_train['fuzz_ratio'].values.reshape(-1,1))
fuzz_ratio_train = normalizer13.transform(X_train['fuzz_ratio'].values.reshape(-1,1))
fuzz_ratio_cv = normalizer13.transform(X_cv['fuzz_ratio'].values.reshape(-1,1))
fuzz_ratio_test = normalizer13.transform(X_test['fuzz_ratio'].values.reshape(-1,1))

from sklearn.preprocessing import Normalizer
normalizer14 = Normalizer()
normalizer14.fit(X_train['fuzz_partial_ratio'].values.reshape(-1,1))
fuzz_partial_ratio_train = normalizer14.transform(X_train['fuzz_partial_ratio'].values.reshape(-1,1))
fuzz_partial_ratio_cv = normalizer14.transform(X_cv['fuzz_partial_ratio'].values.reshape(-1,1))
fuzz_partial_ratio_test = normalizer14.transform(X_test['fuzz_partial_ratio'].values.reshape(-1,1))
```
Stacking all the features together again after vectorization 

```python
from scipy.sparse import hstack
X_tr = hstack((quest1_train,quest2_train,cwc_train,cwcmax_train,csc_min_train,csc_max_train,ctc_min_train,ctc_max_train,last_word_eq_train,first_word_eq_train,abs_len_diff_train,mean_len_train,token_set_ratio_train,token_sort_ratio_train,fuzz_ratio_train,fuzz_partial_ratio_train)).tocsr()
X_cv = hstack((quest1_cv,quest2_cv,cwc_cv,cwcmax_cv,csc_min_cv,csc_max_cv,ctc_min_cv,ctc_max_cv,last_word_eq_cv,first_word_eq_cv,abs_len_diff_cv,mean_len_cv,token_set_ratio_cv,token_sort_ratio_cv,fuzz_ratio_cv,fuzz_partial_ratio_cv)).tocsr()
X_te = hstack((quest1_test,quest2_test,cwc_test,cwcmax_test,csc_min_test,csc_max_test,ctc_min_test,ctc_max_test,last_word_eq_test,first_word_eq_test,abs_len_diff_test,mean_len_test,token_set_ratio_test,token_sort_ratio_test,fuzz_ratio_test,fuzz_partial_ratio_test)).tocsr()
```

Applying logistic regression model on train data 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
#model = LogisticRegression()
#model.fit(X_tr, y_train)
alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr, y_train)
    sig_clf_probs = sig_clf.predict_proba(X_cv)
    cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(y_cv, sig_clf_probs))
```

for alpha = 1e-06
Log Loss : 0.45405238196864106
for alpha = 1e-05
Log Loss : 0.4465275697344375
for alpha = 0.0001
Log Loss : 0.4662334234106437
for alpha = 0.001
Log Loss : 0.5123130640296748
for alpha = 0.01
Log Loss : 0.5607593063636392
for alpha = 0.1
Log Loss : 0.5803403394348411
for alpha = 1
Log Loss : 0.5836654716949035
for alpha = 10
Log Loss : 0.5817714095459336
for alpha = 100
Log Loss : 0.5813038505124879

```python
clf = SGDClassifier(class_weight='balanced', alpha=0.0001, penalty='l2', loss='log', random_state=42)
clf.fit(X_tr, y_train)
test_point_index = 10
no_feature = 500
predicted_cls = sig_clf.predict(X_te[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(X_te[test_point_index]),4))
print("Actual Class :", y_train[test_point_index])
indices = np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:no_feature]
print("-"*50)
```

