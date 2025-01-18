'''
Name: Vishwanath Dattatreya Doddamani
Student ID: A0286188L
NUSNET ID: E1237250
NUS Email: e1237250@u.nus.edu
'''


from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
import nltk
from nltk.tokenize import word_tokenize
import re
import numpy as np

#Start Spark Session
sc = SparkSession \
    .builder \
    .appName("A0286188L Lab2") \
    .getOrCreate()

#Tokenizer for articles.json content
@F.udf(T.ArrayType(T.StringType()))
def tokenize_and_remove_stopwords_udf(text, stopwords):
    tokens = word_tokenize(text) #Uses nltk word_tokenize method
    lowercase_tokens = [token.lower() for token in tokens] #convert tokens to lowercase
    filtered_tokens = [re.match(r'^[a-zA-Z0-9]+', token).group() for token in lowercase_tokens if re.match(r'^[a-zA-Z0-9]+', token)] #matches only alphanumeric characters (takes care of punctuation, special characters, etc.)
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords] #removes stopwords
    return filtered_tokens

#Tokenizer for query.json
@F.udf(T.ArrayType(T.StringType()))
def tokenize_and_remove_stopwords_querydf(text, vocab):
    tokens = word_tokenize(text)
    lowercase_tokens = [token.lower() for token in tokens]
    filtered_tokens = [re.match(r'^[a-zA-Z0-9]+', token).group() for token in lowercase_tokens if re.match(r'^[a-zA-Z0-9]+', token)]
    filtered_tokens = [token for token in filtered_tokens if token in vocab] #Only retian words that are in corpus vocabulary
    return filtered_tokens

#Tokenizer for sentence
def tokenize_and_remove_stopwords_sentencedf(text, vocab):
    tokens = word_tokenize(text)
    lowercase_tokens = [token.lower() for token in tokens]
    filtered_tokens = [re.match(r'^[a-zA-Z0-9]+', token).group() for token in lowercase_tokens if re.match(r'^[a-zA-Z0-9]+', token)]
    filtered_tokens = [token for token in filtered_tokens if token in vocab]
    return filtered_tokens

#Calculate cosine similarity
def compute_cosine_similarity(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0
    else:
        return np.dot(vector1, vector2) / (norm1 * norm2)

#Read input files
articlesDf = sc.read.option("multiline", "true").json('articles.json')
queryDf = sc.read.option("multiline", "true").json('query.json')
stopwordsTxt = sc.read.text('stopwords.txt')

#Make a list of stopwords
stopwordsList = stopwordsTxt.select(F.collect_list('value')).first()[0]

#Pre-process and tokenize the content in articles.json 
articlesDf = articlesDf.withColumn('cleanContent', tokenize_and_remove_stopwords_udf(F.col('content'), F.array([F.lit(x) for x in stopwordsList])))

#Calculate term frequency, map to ((article_id, word), 1), reducer will count all the occurences
articles_df_rdd = articlesDf.rdd.flatMap(lambda row: [((row['id'], word), 1) for word in row['cleanContent']])
articles_df_cv = articles_df_rdd.reduceByKey(lambda x, y: x + y)

#Calculate idf. Map to (word, 1) and then reduce, here word will be counted only once per document
articles_df_idf_rdd = articlesDf.rdd.flatMap(lambda row: [(word, 1) for word in set(row['cleanContent'])])
articles_df_idf = articles_df_idf_rdd.reduceByKey(lambda x, y: x + y)

D = articlesDf.count() #Calculate total number of articles (documents)
articles_df_idf = articles_df_idf.mapValues(lambda x: np.log10(D/x)) #Calculate idf for each word


articles_df_tf = articles_df_cv.mapValues(lambda x: np.log10(1 + x)) #Convert tf to (1 + log(tf))
articles_df_tf = articles_df_tf.map(lambda x: (x[0][1], (x[0][0], x[1]))) #map to (word, (id, tf_score)) so that it can be joined with idf rdd

articles_df_tfidf = articles_df_tf.join(articles_df_idf) #join articles tf rdd to articles idf rdd
articles_df_tfidf = articles_df_tfidf.mapValues(lambda x: (x[0][0], x[0][1]*x[1])) #Calculate tf * idf

articles_df_tfidf = articles_df_tfidf.map(lambda x: ((x[1][0], x[0]), x[1][1])) 

articles_df_tfidf = articles_df_tfidf.map(lambda x: (x[0][0], (x[0][1], x[1]))) #Re-map it to (id, (word, tf-idf_score))


#Store idf of words in a dictionary for future use
words_rdd = articles_df_idf.map(lambda x: x[0]) 
vocabulary = words_rdd.collect() #Create list of vocabulary words
vocab_len = len(vocabulary)
words_idf = dict(articles_df_idf.collect()) #Create idf dictionary {word : idf_score}
vocabulary_index = {word: index for index, word in enumerate(vocabulary)} #index of all vocabulary words to make computing tf-idf vector efficient

#Calculate tf-idf vector for all articles (documents)
articles_df_tf_idf_group = articles_df_tfidf.groupByKey().collect()
articles_tf_idf_dict = {}
for key, values in articles_df_tf_idf_group:
    tf_vector = np.zeros(vocab_len) #Len of a vector will be len of vocabulary
    for value in values:
        tf_vector[vocabulary_index[value[0]]] = value[1] #fill vector values according to word index
    articles_tf_idf_dict[key] = tf_vector #Assign vector to word


#Do the processing steps for queries as done for articles
queryDf = queryDf.withColumn('cleanContent', tokenize_and_remove_stopwords_querydf(F.col('query'), F.array([F.lit(x) for x in vocabulary])))
query_df_rdd = queryDf.rdd.flatMap(lambda row: [((row['id'], word), 1) for word in row['cleanContent']])
query_df_cv = query_df_rdd.reduceByKey(lambda x, y: x + y)
query_df_tf = query_df_cv.mapValues(lambda x: np.log10(1 + x))
query_df_tf = query_df_tf.map(lambda x: (x[0][1], (x[0][0], x[1])))
query_df_tfidf = query_df_tf.join(articles_df_idf) #Use articles idf to compute idf of queries
query_df_tfidf = query_df_tfidf.mapValues(lambda x: (x[0][0], x[0][1]*x[1]))
query_df_tfidf = query_df_tfidf.map(lambda x: ((x[1][0], x[0]), x[1][1]))
query_df_tfidf = query_df_tfidf.map(lambda x: (x[0][0], (x[0][1], x[1])))

#Calculate tf-idf vector for queries using similar logic as used in articles
query_df_tf_idf_group = query_df_tfidf.groupByKey().collect()
query_tf_idf_dict = {}
for key, values in query_df_tf_idf_group:
    tf_vector = np.zeros(vocab_len)
    for value in values:
        tf_vector[vocabulary_index[value[0]]] = value[1]
    query_tf_idf_dict[key] = tf_vector


#Find top relevant article for a query
top_similar_keys = {}
final_result = []

for query_key, query_vector in query_tf_idf_dict.items():
    cosine_similarities = {}
    for key, vector in articles_tf_idf_dict.items():
        cosine_similarities[key] = compute_cosine_similarity(query_vector, vector)
    sorted_keys_by_similarity = sorted(cosine_similarities, key=cosine_similarities.get, reverse=True)
    top_key = sorted_keys_by_similarity[0] #Get article with highest relevance (cosine similarity) score
    top_similar_keys[query_key] = {top_key: cosine_similarities[top_key]}

    #Calculate relevance score and find top relevant sentence
    top_row = articlesDf.filter(articlesDf['id'] == top_key).select('content').first()
    top_content = top_row['content']
    top_content_sent = nltk.sent_tokenize(top_content) #Uses nltk sentence tokenizer to split article into sentences
    cosine_similarities_sent = {}
    for index, sentence in enumerate(top_content_sent):
        words_split = tokenize_and_remove_stopwords_sentencedf(sentence, vocabulary)
        words_cv = {}
        for word in words_split:
            words_cv[word] = np.log10(1 + words_split.count(word)) * words_idf[word]
        words_tfidf = np.zeros(vocab_len)
        for word in words_cv.keys():
            words_tfidf[vocabulary_index[word]] = words_cv[word]
        cosine_similarities_sent[index] = compute_cosine_similarity(query_vector, words_tfidf)
    
    sorted_keys_by_similarity_sent = sorted(cosine_similarities_sent, key=cosine_similarities_sent.get, reverse=True)
    top_sent_key = sorted_keys_by_similarity_sent[0]
    top_sent = top_content_sent[top_sent_key]

    #Store the results to output
    final_result.append([query_key, top_key, cosine_similarities[top_key], top_sent, cosine_similarities_sent[top_sent_key]])

max_len_array = sorted(final_result, key = lambda x:len(x[3]), reverse=True)
max_len = len(max_len_array[0][3])
final_result = sorted(final_result, key = lambda x : x[0])

#Writing Output to a file
with open('output.txt', 'w') as file:
    file.write(f"{'query id':<10}{'article id':<15}{'article score':<15}{'sentence':<{max_len+5}}{'sentence score':<10}\n")
    for query_id, article_id, article_score, sentence, sentence_score in final_result:
        file.write(f"{str(query_id):<10}{str(article_id):<15}{str(np.round(article_score, decimals = 6)):<15}{str(sentence):<{max_len+5}}{str(np.round(sentence_score, decimals = 6)):<10}\n")


sc.stop() #Stop Spark session