from flask import Flask, request, jsonify
from flask import Flask, render_template, request
from main import model, bag_of_words, words, labels, data
import numpy as np
import random
import requests

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response_intent():
    userText = request.args.get('msg')
    results = model.predict([bag_of_words(userText, words)])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return str(random.choice(responses))
    else:
        return "I didn't get that please try again"
    

if __name__ == "__main__":
    app.run(debug=True)
