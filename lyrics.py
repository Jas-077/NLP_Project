import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from keras.models import load_model
import glob 
import nltk
import re
nltk.download('punkt')

def generate_text(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []
    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating = 'pre')
        y_predict = model.predict(encoded)
        #print(np.argmax(y_predict))
        word_index = np.argmax(y_predict)
        
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == word_index:
                predicted_word = word
                break
        seed_text = seed_text+ ' ' + predicted_word
        text.append(predicted_word)
        
    return ' '.join(text)

def pre(seed, no_words):
    
    data = ""
    folder_path = "data/*"
    files = glob.glob(folder_path)
    for f in files:
      cs_v = pd.read_csv(f)
      cs_v.dropna(inplace=True)
      #print(f"File is: {f}")
      for r in range(len(cs_v)):
          
            #print(f"Line no: {r}")
          try:
              
              lyr = cs_v['Lyric'][r]
              lyr = re.sub("'d"," would",lyr)
              lyr = re.sub("n't"," not",lyr)
              lyr = re.sub("'m"," am",lyr)
              lyr = re.sub("'s"," is",lyr)
              lyr = re.sub("'ve"," have",lyr)
              data+=lyr
          except:
        #print(f"Error in {r}")
            pass 
    
    y = nltk.word_tokenize(data)
    corpus= []
    for w in y:
        x = w.lower()
        x = re.sub('[^a-z0-9]','',x)
        if x:
            corpus.append(x)
            
    sent_len = 50 + 1
    all_sent = []
    total_words = len(corpus)
    for i in range(total_words-sent_len):
        sent = corpus[i:sent_len+i]
        z = ' '.join(sent)
        all_sent.append(z)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sent)
    sequences = tokenizer.texts_to_sequences(all_sent)
    sequences = np.array(sequences) 
        
    voc_size = len(tokenizer.word_index) +1
    X = []
    Y =[]
    X, Y = sequences[:,:-1] , sequences[:, -1]
    Y = np_utils.to_categorical(Y, num_classes= voc_size)
    #print(X[:10])
    #print(Y[:10])
    embedding_vector_features = 50
    #print(X.shape)
    #print(Y.shape)
    #print(voc_size)
    model = Sequential()
    model = load_model('Model2_150ep')
    #print(model.summary())
    seed = seed
    no_words = int(no_words)
    txt = generate_text(model, tokenizer, 50, seed, no_words)
    #print("Seed txt: ",seed)
    #print("Predicted: ",txt)
    return txt


    