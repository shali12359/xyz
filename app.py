from flask import Flask, request, jsonify
import random
import flask
from Spotting_system import keyword_spotting_service
import os
import werkzeug

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # imagefile = flask.request.files['file']
    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    # print("\nReceived image File name : " + imagefile.filename)
    # imagefile.save(filename)

    # GET AUDIO FILE & SAVE
    audio = request.files['file']
    fileName = str(random.randint(0, 100000))
    audio.save(fileName)

    # INVOKE SPOTTING SYSTEM
    kss = keyword_spotting_service()

    # MAKE PREDICTION
    predicted_keyword = kss.predict(fileName)

    # REMOVE AUDIO FILE
    os.remove(fileName)

    # SEND BACK PREDICTION
    data = {'Keyword': predicted_keyword}

    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=9696)