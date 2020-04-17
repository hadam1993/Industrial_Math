#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
import re
warnings.filterwarnings('ignore')
from pattern.en import spelling


# In[2]:


import re


# In[3]:


def reduce_lengthening(text):
    pattern = re.compile(r"([a-zA-Z])\1{2,}")
    return pattern.sub(r"\1\1", text)


# In[58]:


def remove_extra_spaces(text):
    pattern = re.compile(r"(\s)\1{2,}")
    return pattern.sub(r"\1", text)


# In[66]:


text = "       asdrfr    "
print(remove_extra_spaces(text))


# In[118]:


from pattern.en import suggest
sentence = "The score has gone finalllllll"
word = "loool"
word_wlf = reduce_lengthening(word) #calling function defined above
print(word_wlf) #word lengthening isn't being able to fix it completely

correct_word = suggest(word_wlf) 
print(correct_word)


# In[4]:


import spacy

def lemmatize(tweet, nlp):
    doc = nlp(tweet)
    newSentence = " ".join([token.lemma_ for token in doc])
    return(newSentence)


# In[5]:


def spellCheck(tweet):
    tokenz = word_tokenize(tweet)
    newSentence = ""
    for token in tokenz:
        correct_word = suggest(token)
        newWord = correct_word[0][0]
        newSentence += newWord + " "
    newSentence = newSentence[:-1]
    return(newSentence)


# In[124]:


import wordninja
def splitWords(tweet):
    doc = nlp(tweet)
    newSentence = " ".join([" ".join(wordninja.split(str(token))) for token in doc])
    return(newSentence)


# In[125]:


import nltk


# In[2]:


dataset = pd.read_csv('./data/Kaggle/train.csv',delimiter=',',                           names=['id','keyword','location', 'text','target'])
dataset = dataset.drop(0)
dataset.head()


# In[127]:


# Drop Id, Keyword, Location
dataset = dataset.drop(labels=['id', 'keyword','location'], axis=1)
dataset.head()


# In[128]:


import string
#from string import maketrans
def clean(tweet,nlp):
    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
    tweet = re.sub(r"åÇ", "", tweet)
    tweet = re.sub(r"å£3million", "3 million", tweet)
    tweet = re.sub(r"åÀ", "", tweet)
    tweet = re.sub(r"amp", "and", tweet)
    tweet = re.sub(r"\n", "", tweet)
    tweet = re.sub(r"\r", "", tweet)
    tweet = tweet.lower()
    tweet = reduce_lengthening(tweet)
    
    tweet = re.sub(r"x\d+", "", tweet) 
    tweet = re.sub(r"\d", "", tweet) 
    tweet = re.sub(r"\u0089ã¢", "", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)
    # Remove http
    tweet = re.sub(r"http[^\s]+","", tweet)
    tweet = re.sub(r"http","", tweet)
    tweet = re.sub(r"youtube","", tweet)
    # Remove @abc
    tweet = re.sub(r"@[^\s]+", "", tweet)
    tweet = tweet.translate(str.maketrans('','',string.punctuation))
    tweet = lemmatize(tweet,nlp)
    tweet = remove_extra_spaces(tweet)
    tweet = re.sub(r"^\s+","", tweet)
    tweet = splitWords(tweet)
    
    return tweet


# In[129]:


# Drop first row
dataset = dataset.drop(index=0)
# Clean data
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
dataset['text_cleaned'] = dataset['text'].apply(lambda s : clean(s,nlp))


# In[130]:


dataset.head()


# In[131]:


dataset['text_cleaned'] = dataset['text_cleaned'].drop_duplicates()
dataset['text_cleaned'].replace('', np.nan, inplace=True)
dataset.dropna(subset=['text_cleaned'], inplace=True)


# In[132]:


dataset.head()


# In[133]:


dataset['text_cleaned'].to_csv("dataset_cleaned2.csv")


# In[ ]:


from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()
dataset['tokenized'] = dataset['text_cleaned'].apply(tt.tokenize)
dataset.head()


# In[ ]:


wordList = sum(dataset['tokenized'].values,[])
wordList


# In[ ]:


freDist = nltk.FreqDist(wordList)


# In[ ]:


freDist


# In[ ]:


freDist.keys()


# ### Try Tokenized BERT Embeding and Create Model

# In[3]:


def pad_token_list(sample):
    # Find the sentence with the max length
    max_len = 0
    for token_list in sample:
        if len(token_list) > max_len:
            max_len = len(token_list)
    # Adjust every sentence to the same length
    padded = np.array([token_list + [0]*(max_len-len(token_list)) for token_list in sample])
    return padded, max_len


# In[4]:


def get_embeddings_from_sample(sample, model):
    # Pad sample data:
#     sample = pad_token_list(sample)
    # Define mask from data: - 0 token entry     -> padding, set mask entry to 0
    #                        - non-0 token entry -> valid word, set mask entry to 1
    mask = np.where(sample != 0, 1, 0)
    
    # Create tensor objects from numpy arrays
    input_ids = torch.tensor(sample).long()
    attention_mask = torch.tensor(mask).long()

    # Use BERT model to get embeddings
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    # Extract [CLS] embedding for each sample as numpy array to be used for classification task
    features = last_hidden_states[0][:,0,:].numpy()
    return features, mask


# In[5]:


clean_dataset = pd.read_csv('dataset_cleaned2.csv',header=None)


# In[6]:


clean_dataset.head()
#clean_dataset['target'] = dataset.loc()


# In[20]:


dataset.loc[1]


# In[7]:


clean_dataset['target'] = dataset['target'].loc[clean_dataset[0]].tolist()


# In[8]:


dataset = clean_dataset
dataset.columns = ['ID','text_cleaned','target']
#clean_dataset[0]
dataset.head()


# In[9]:


sample_size = 4000
random_sample = dataset.sample(n=sample_size)
random_sample.shape


# In[10]:


val_set = dataset.loc[dataset.index.difference(random_sample.index)]
val_set.shape


# In[11]:


val_sample_size = 1500
val_random_sample = val_set.sample(n=val_sample_size)
val_random_sample.shape
test_set = val_set.loc[val_set.index.difference(val_random_sample.index)]
test_set.shape


# In[12]:


# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# In[13]:


sample_tokenized = random_sample['text_cleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#sample_tokenized2 = random_sample2['text_cleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
val_random_sample_tokenized = val_random_sample['text_cleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#test_tokenized = test_set['text_cleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[14]:


sample_padded, sample_len = pad_token_list(sample_tokenized.values)
val_padded, val_len = pad_token_list(val_random_sample_tokenized.values)
#sample_padded2, sample_len2 = pad_token_list(sample_tokenized2.values)
#test_padded, test_len = pad_token_list(test_tokenized.values)


# In[15]:


sample_features, mask = get_embeddings_from_sample(sample_padded, model)
val_features, mask = get_embeddings_from_sample(val_padded, model)


# In[16]:


sample_features.shape


# In[17]:


# Create cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[46]:


train_features_tensor = torch.tensor(np.asarray(sample_features))
train_features_tensor = train_features_tensor.to(device)
train_labels_tensor =  torch.FloatTensor(np.asarray(random_sample['target']).astype(np.float))
train_labels_tensor = train_labels_tensor.to(device)

val_features_tensor = torch.tensor(np.asarray(val_features))
val_features_tensor = val_features_tensor.to(device)
val_labels_tensor =  torch.FloatTensor(np.asarray(val_random_sample['target']).astype(np.float))
val_labels_tensor = val_labels_tensor.to(device)


# In[19]:


# Define neural network class to be trained
# Structure:
# input -> fc1 -> sigmoid -> out -> log_softmax
import torch.nn as nn
import torch.nn.functional as F
class Shallow_Network(nn.Module):
    def __init__(self):
        super(Shallow_Network,self).__init__()
        self.fc1 = nn.Linear(768,1000)
        self.out = nn.Linear(1000,2)
    def forward(self,input):
        # Take input, feed through fc1 layer,
        # then apply activation function to it
        x = F.sigmoid(self.fc1(input))
        # Take output of sigmoid, input into out layer,
        # and apply log_softmax function
        return (F.log_softmax(self.out(x),dim=1))


# In[50]:


class Medium_Network(nn.Module):
    def __init__(self):
        super(Medium_Network,self).__init__()
        self.fc1 = nn.Linear(768,1000)
        self.fc2 = nn.Linear(1000,5000)
        self.fc3 = nn.Linear(5000,1000)
        self.out = nn.Linear(1000,1)
    def forward(self,input):
        # Take input, feed through fc1 layer,
        # then apply activation function to it
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.out(x))
        return(x)
        # Take output of sigmoid, input into out layer,
        # and apply log_softmax function
        #return (F.log_softmax(self.out(x),dim=1))


# In[84]:


# Create neural network object
#net = Shallow_Network()
net = Medium_Network()
net = net.to(device)


# In[85]:


import torch.optim as optim
#Create an stochastic gradient descent optimizer
adam = optim.Adam(net.parameters(), lr=0.001)
#loss_func = nn.NLLLoss()
loss_func = nn.BCELoss()
loss_func = loss_func.to(device)


# In[83]:


def accuracy(net,features,labels):
    # Get classification probabilities from hidden state array
    # And apply Softmax
    with torch.no_grad():
        probs = net(features)
        #softprobs = F.softmax(probs)
    # Get most likely class and its index for each sample point
    #values, indices = torch.max(softprobs,1)
    values = torch.round(probs)
    # Calculate number of sample points where prediction failed
    #nums = torch.sum(torch.abs(labels-indices)).detach().cpu().numpy()
    nums = torch.sum(torch.abs(torch.t(torch.round(probs)) - labels))
    #nums = torch.sum(torch.abs(labels-values)).detach().cpu().numpy()
    # Number of correct predictions
    numcorrect = len(labels)-(nums+0)
    # Accuracy of prediction
    accuracy = numcorrect/len(labels)
    return(accuracy)


# In[86]:


# Train network
cnt = 0
average_losses = []
average_val_losses = []
acc = []
cur_loss = []
min_validation = 10000.0
min_val_epoch = 0
for epoch in range(400):
    net.train()
    #zero the gradient
    adam.zero_grad()
    #Get output of network
    probs = net(train_features_tensor)
    #compute loss
    loss = loss_func(probs,train_labels_tensor)
    #compute the backward gradient and move network in that direction
    loss.backward()
    adam.step()
    #gather loss
    cur_loss.append(loss.detach().cpu().numpy())
    print("epoch ",epoch)
    print("training loss: ", np.mean(cur_loss))
    net.eval()
    probs_val = net(val_features_tensor)
    loss_val = loss_func(probs_val,val_labels_tensor)
    print("validation loss: ", np.mean(loss_val.detach().cpu().numpy()))
    print("validation accuracy: ", accuracy(net,val_features_tensor,val_labels_tensor))
    #Save model if validation is min
    if min_validation > np.mean(loss_val.detach().cpu().numpy()):
        min_validation = np.mean(loss_val.detach().cpu().numpy())
        min_val_epoch = epoch
        torch.save(net.state_dict(), './net_parameters_kaggle.pth')
    


# In[80]:


torch.t(torch.round(probs[0:5]))


# In[81]:


val_labels_tensor[0:5]


# In[82]:


torch.sum(torch.abs(torch.t(torch.round(probs[0:5])) - val_labels_tensor[0:5]))


# In[ ]:


min_val_epoch


# In[ ]:


net = Shallow_Network()
checkpoint = torch.load('./net_parameters_kaggle.pth')
net.load_state_dict(checkpoint)
net = net.to(device)
net.eval()


# In[ ]:


probs_val = net(val_features_tensor)
loss_val = loss_func(probs_val,val_labels_tensor)
print("validation loss: ", np.mean(loss_val.detach().cpu().numpy()))


# In[ ]:


print(accuracy(net,val_features_tensor,val_labels_tensor))


# In[ ]:


test_set = val_set.loc[val_set.index.difference(val_random_sample.index)]
test_random_sample_tokenized = test_set['text_cleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
test_padded, test_len = pad_token_list(test_random_sample_tokenized.values)
test_features, mask = get_embeddings_from_sample(test_padded, model)
test_features_tensor = torch.tensor(np.asarray(test_features))
test_features_tensor = test_features_tensor.to(device)
test_labels_tensor =  torch.tensor(np.asarray(test_set['target']).astype(np.int))
test_labels_tensor = test_labels_tensor.to(device)


# In[ ]:


print(accuracy(net,test_features_tensor,test_labels_tensor))


# Baseline only html tag and special character cleaning yielded 81.8% 

# In[29]:


dataset.shape


# In[30]:


dataset = dataset.drop_duplicates(subset='text_cleaned')


# In[33]:


dataset.shape


# In[34]:


dataset.head()


# In[ ]:


ã¢
&gt;
&amp;

