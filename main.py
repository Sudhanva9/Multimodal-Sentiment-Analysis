# import os
# from flask import Flask, request, render_template
# from deepface import DeepFace
#
# app = Flask(__name__)
#
# # Define the folder to store uploaded files
# UPLOAD_FOLDER = 'static'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
#
# @app.route('/')
# def index():
#     return render_template('multi.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'image' not in request.files:
#         print("no file found ----------")
#         return "No file part"
#
#     file = request.files['image']
#     print(file.filename)
#
#     if file.filename == '':
#         return "No selected file"
#
#     # Save the file to the defined folder
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
#     return "File uploaded successfully"
#
# # Define an API endpoint to return the data
# @app.route('/data')
# def get_data():
#     # Your data
#     # file = request.files['image']
#     data = DeepFace.analyze(img_path='static//fear.png',
#                             actions=['emotion'])
#
#     print(data)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
#

import os
import json
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from deepface import DeepFace

from description import get_image_descriptions
from text import emotion_classifier
import openai

# from text import predict_sentiment, predict_emotion
# from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'static'  # Save uploads directly in the static folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def text():
    return render_template('text.html')

# ===================================================== TEXT.IO =====================================================

#emotion = emotion_classifier("I appreciate it, that's good to know. I hope I'll have to apply that knowledge one day")
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            # Call emotion_classifier function to get emotion
            emotion = emotion_classifier(text)
            # Return the emotion as JSON
            print(emotion)
            return jsonify({'label': emotion['label'], 'score': emotion['score']})
        else:
            return jsonify({'error': 'Text not provided'})


# ===================================================== MULTI.IO =====================================================

@app.route('/multi')
def multi():
    return render_template('multi.html')

# @app.route('/')
# def index():
#     return render_template('multi.html')
    
import re

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        # Handle POST request here
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['image']
        text = request.form['text']
        print('text', text)

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the uploaded file
            file.save(file_path)

            # Get image descriptions using description.py
            descriptions = get_image_descriptions([file_path], text)[0]
            print(descriptions)

            # ===================================================== GP =====================================================

            # Perform analysis on the uploaded image
            data = DeepFace.analyze(img_path='static//'+filename, actions=['emotion', 'age', 'gender', 'race'])

            # Extracting description
            description = descriptions.split("Text present in the image:")[0].strip()

            # Extracting text present in the image
            # text_in_image = re.findall(r'-\s*([^\n]+)', given_string.split("Text present in the image:")[1].strip().split('Dominant Emotion')[0].strip())
            text_in_image = ', '.join(re.findall(r'-\s*([^\n]+)',
                                                 descriptions.split("Text present in the image:")[1].strip().split(
                                                     'Dominant Emotion')[0]))
            # Extracting dominant emotion
            dominant_emotion = re.findall(r'-\s*([^\n:]+):\s*(\d+%)', descriptions.split("Dominant Emotion:")[1])[0]

            # Extracting other emotions
            other_emotions = dict(re.findall(r'-\s*([^\n:]+):\s*(\d+%)', descriptions.split("Other Emotions:")[1]))

            # Printing the parsed data
            print("Description:", description)
            print("Text present in the image:", text_in_image)
            print("Dominant Emotion:", dominant_emotion[0])
            print("Other Emotions:")
            data[0]['emotion'] = {}
            for emotion, percentages in other_emotions.items():
                print(emotion, percentages)
                data[0]['emotion'][emotion] = percentages

            data[0]['descriptions'] = description
            data[0]['dominant_emotion'] = dominant_emotion[0]
            data[0]['text_in_image'] = text_in_image

            # ===================================================== GP =====================================================

            print(data)

            # Return the data as JSON
            return jsonify(data)

            # Return the sentiment and emotion results
            return jsonify({'sentiment': sentiment_result, 'emotion': emotion_result})

    elif request.method == 'GET':
        # Handle GET request here
        # You can return some default data if needed
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Perform analysis on the uploaded image
            mlData = DeepFace.analyze(img_path='static//'+filename, actions=['emotion'])

            print("get-----", mlData)

            # Return the data as JSON
            return jsonify(mlData)
        return jsonify({'message': 'GET request received for /data'})



    # If the request method is not GET or POST
    return jsonify({'error': 'Invalid request method'})




if __name__ == '__main__':
    app.run(debug=True)

