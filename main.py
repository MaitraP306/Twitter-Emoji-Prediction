#The following is not completed I am still working on it 

# In[39]:


import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN,Embedding
from emoji import core
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# In[40]:


data=pd.read_csv("train.csv")
mapping=pd.read_csv("mapping.csv")
mapping=mapping.to_dict()["emoticons"]
data=data.drop('Unnamed: 0', axis=1)


# In[41]:


Xtrain=data["TEXT"].values
Ytrain=data["Label"].values


# In[42]:


file=open("glove.6B.100d.txt","r",encoding="utf8")
content=file.readlines()
file.close()


# In[43]:


embeddings={}
for line in content:
    line=line.split()
    embeddings[line[0]]=np.array(line[1:],dtype=float)


# In[44]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(Xtrain)
word2index=tokenizer.word_index
Xtrain=tokenizer.texts_to_sequences(Xtrain)


# In[45]:


def padding(data):
    L=[]
    for j in range (0,len(data)):
        L.append(len(data[j]))
    n=max(L)
    for j in range (0,len(data)):
        for k in range (0,n-L[j]):
            data[j].append(0)
    return data,n
Xtrain,maxlen=padding(Xtrain)


# In[ ]:





# In[46]:


Ytrain=to_categorical(Ytrain)


# In[47]:


def intialize_emb_matrix(file):
    embedding_matrix = {}
    for line in file:
        values = line.split()
        word = values[0]
        embedding = np.array(values[1:], dtype='float64')
        embedding_matrix[word] = embedding

    return embedding_matrix 


# In[48]:


embedding_matrix=intialize_emb_matrix(content)


# In[ ]:





# In[50]:


model=Sequential([
    Embedding(input_dim = len(word2index)+1,
              output_dim = 168,
              input_length = maxlen,
              weights = [embedding_matrix],
              trainable = False
             ),
])
model.add(LSTM(units = 256, return_sequences=True, input_shape = (168,100)))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[75]:


res = model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=10, verbose=2)
