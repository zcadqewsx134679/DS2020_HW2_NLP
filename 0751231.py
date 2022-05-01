# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:10:13 2020

@author: USER
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D

df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

label = []
label_test = []
for i in range(df.shape[0]):
    label.append(df['Label'][i])
label = np.array(label)
label-=3

embeddings_index = {}
f = open('glove.6B.100d.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

sent = []
data = []
data_test = []
for i in range(df.shape[0]):
    sent=[(df['Headline'][i])]
    stopwords=[",","the"]
    list=[word for word in word_tokenize(sent[0]) if word not in stopwords]
    pos = nltk.pos_tag(list)
    wordnet_pos = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos:
        if tag.startswith('J'):
            wordnet_pos.append(wordnet.ADJ)
        elif tag.startswith('V'):
            wordnet_pos.append(wordnet.VERB)
        elif tag.startswith('N'):
            wordnet_pos.append(wordnet.NOUN)
        elif tag.startswith('R'):
            wordnet_pos.append(wordnet.ADV)
        else:
            wordnet_pos.append(wordnet.NOUN)

    tokens = [lemmatizer.lemmatize(pos[n][0], pos=wordnet_pos[n]) for n in range(len(pos))]
    data.append(tokens)
    
for j in range(df2.shape[0]):
    sent=[(df2['Headline'][j])]
    stopwords=[",","the"]
    list=[word for word in word_tokenize(sent[0]) if word not in stopwords]
    pos = nltk.pos_tag(list)
    wordnet_pos = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos:
        if tag.startswith('J'):
            wordnet_pos.append(wordnet.ADJ)
        elif tag.startswith('V'):
            wordnet_pos.append(wordnet.VERB)
        elif tag.startswith('N'):
            wordnet_pos.append(wordnet.NOUN)
        elif tag.startswith('R'):
            wordnet_pos.append(wordnet.ADV)
        else:
            wordnet_pos.append(wordnet.NOUN)

    tokens = [lemmatizer.lemmatize(pos[n][0], pos=wordnet_pos[n]) for n in range(len(pos))]
    data_test.append(tokens)
    
tokenizer_train = Tokenizer(nb_words=None)
tokenizer_test = Tokenizer(nb_words=None)
tokenizer_train.fit_on_texts(data)
tokenizer_test.fit_on_texts(data_test)

data_total = data + data_test
tokenizer_total = Tokenizer(nb_words=None)
tokenizer_total.fit_on_texts(data_total)

sequences = tokenizer_train.texts_to_sequences(data)
sequences_test = tokenizer_train.texts_to_sequences(data_test)

word_index = tokenizer_train.word_index
word_index_test = tokenizer_test.word_index
word_index_total = tokenizer_total.word_index
print('Found %s unique tokens.' % len(word_index))
print('Found %s unique tokens.' % len(word_index_test))
print('Found %s unique tokens.' % len(word_index_total))

data1 = pad_sequences(sequences, maxlen=40)
data_test1 = pad_sequences(sequences_test, maxlen=40)

embedding_matrix = np.zeros((len(word_index_total) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index_total) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=40,
                            trainable=False)

x_train = data1
x_test = data_test1
y_train = label

sequence_input = Input(shape=(40,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(filters=100, kernel_size=5,strides=1, padding="causal",activation="relu")(embedded_sequences)
x = Conv1D(filters=200, kernel_size=5,strides=1, padding="causal",activation="relu")(x)
x = MaxPooling1D(pool_size=5, strides=1, padding='valid')(x)
x = SpatialDropout1D(0.2)(x)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
x = BatchNormalization()(x)
x = Dense(256)(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = Dense(256)(x)
x = LeakyReLU(0.2)(x)
output = Dense(1)(x)
model = Model(sequence_input,output)
model.compile(loss='mse',optimizer='adam')
model.summary()

history = model.fit(x_train,y_train,shuffle= True,validation_split=0.2,epochs=15,batch_size=128)
y_predict = model.predict(x_test)
y_predict +=3
col_name=df2.columns.tolist()
df2['Label']=y_predict
df2.to_csv('0751231.csv',index=0,columns = ['ID','Label'])
