import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from time import sleep
import sys
import os
from welcome import Welcome

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def writer(words):
    for char in words:
        if char == ' ' or char == ',':
            sleep(0.09)
            sys.stdout.write(char)
            sys.stdout.flush()
        elif char == '.'or char == '\n':
            sleep(0.2)
            sys.stdout.write(char)
            sys.stdout.flush()
        else:
            sleep(0.05)
            sys.stdout.write(char)
            sys.stdout.flush()

Welcome()

os.system("cls")
print('''\u001b[34m      
    ░██████╗██████╗░███╗░░░███╗
    ██╔════╝██╔══██╗████╗░████║
    ╚█████╗░██████╔╝██╔████╔██║
    ░╚═══██╗██╔══██╗██║╚██╔╝██║
    ██████╔╝██║░░██║██║░╚═╝░██║
    ╚═════╝░╚═╝░░╚═╝╚═╝░░░░░╚═╝
''')
print("\u001b[35m______________________________________________________________________________________________________________._._._..._-_-_-_-_--_--_--_--.-~'`\n|")
while True:
    sentence = input("\u001b[35m|\u001b[33m You: \u001b[37m")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"\u001b[35m|\u001b[36m SRM: \u001b[37m",end="")
                writer(random.choice(intent['responses']))
                print("\n",end="")
    else:
        print(f"\u001b[35m|\u001b[36m SRM: \u001b[37m",end="")
        writer("sorry...I do not understand...")
        print("\n",end="")