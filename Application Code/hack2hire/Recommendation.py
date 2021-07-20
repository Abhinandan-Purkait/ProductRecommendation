#import dependencies
import nltk
import time
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)
words = []
labels = []
docs_x = []
docs_y = []


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)



dataset=[]
for i in range(0,len(output)):
    dataset.append([training[i],output[i]])




random.shuffle(dataset)



training=[]
output=[]
for x,y in dataset:
    training.append(x)
    output.append(y)
training = numpy.array(training)
output = numpy.array(output)

'''
print(training)
print(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=10, batch_size=1, show_metric=True)
model.save("model.tflearn")
'''
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('./model.tflearn', weights_only=True)
type(model)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    inp=input("Enter String")

    results = model.predict([bag_of_words(inp, words)])
    print(results)
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    print(tag)


if __name__ == "__main__":
    chat()
