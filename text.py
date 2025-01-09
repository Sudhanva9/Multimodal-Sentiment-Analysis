import os

import palette
from tensorboard.notebook import display

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import torch
# from transformers import pipeline


tqdm.pandas()

import torch
import torch.nn as nn


def emotion_classifier(message):
    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Preparing for TPU usage
    # import torch_xla
    # import torch_xla.core.xla_model as xm
    # device = xm.xla_device()

    df = pd.read_csv('kaggle/input/go-emotions-google-emotions-dataset/go_emotions_dataset.csv')
    emotions = set(df.columns[3:])


    positive = {'admiration','amusement','approval','caring','desire','excitement','gratitude','joy','love','optimism','pride','relief'}
    negative = {'sadness','fear','embarrassment','disapproval','disappointment','annoyance','anger','nervousness','remorse','grief','disgust'}
    ambiguous = {'realization','surprise','curiosity','confusion','neutral'}

    df_emotion = pd.DataFrame()
    df_emotion['emotion'] = list(emotions)
    df_emotion['group'] = ''
    df_emotion['group'].loc[df_emotion['emotion'].isin(positive)] = 'positive'
    df_emotion['group'].loc[df_emotion['emotion'].isin(negative)] = 'negative'
    df_emotion['group'].loc[df_emotion['emotion'].isin(ambiguous)] = 'ambiguous'

    import preprocessor
    import contractions


    def clean_text(text):
        re_number = re.compile('[0-9]+')
        re_url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        re_tag = re.compile('\[[A-Z]+\]')
        re_char = re.compile('[^0-9a-zA-Z\s?!.,:\'\"//]+')
        re_char_clean = re.compile('[^0-9a-zA-Z\s?!.,\[\]]')
        re_punc = re.compile('[?!,.\'\"]')

        text = re.sub(re_char, "", text)  # Remove unknown character
        text = contractions.fix(text)  # Expand contraction
        text = re.sub(re_url, ' [url] ', text)  # Replace URL with number
        text = re.sub(re_char_clean, "", text)  # Only alphanumeric and punctuations.
        # text = re.sub(re_punc, "", text) # Remove punctuation.
        text = text.lower()  # Lower text
        text = " ".join([w for w in text.split(' ') if w != " "])  # Remove whitespace

        return text
    # ======================================================================================================
    # def predict_sentiment(text):
    #     # Initialize VADER sentiment intensity analyzer
    #     analyzer = SentimentIntensityAnalyzer()
    #
    #     # Analyze sentiment of the text
    #     sentiment_scores = analyzer.polarity_scores(text)
    #
    #     # Get the sentiment label based on the compound score
    #     if sentiment_scores['compound'] >= 0.05:
    #         sentiment = "positive"
    #     elif sentiment_scores['compound'] <= -0.05:
    #         sentiment = "negative"
    #     else:
    #         sentiment = "neutral"
    #
    #     return sentiment
    #
    # emotion_classifier = pipeline("sentiment-analysis")
    #
    # # Function to predict emotion
    # def predict_emotion(text):
    #     # Use the pre-trained model to predict emotion
    #     result = emotion_classifier(text)
    #     # Extract the predicted emotion label
    #     emotion_label = result[0]['label']
    #     return emotion_label


    # ======================================================================================================


    data = pd.read_csv('kaggle/input/go-emotions-google-emotions-dataset/go_emotions_dataset.csv')

    data.drop('id', inplace=True, axis=1)
    data.drop('example_very_unclear', inplace=True, axis=1)

    data["cleaned_text"] = data["text"].progress_apply(clean_text)


    data['emotion'] = (data.iloc[:, 1:] == 1).idxmax(1)

    # Reorganizing DataFrame (for sanity)
    data = data[ ['cleaned_text', 'emotion'] + [ col for col in data.columns if col not in ['text', 'cleaned_text', 'emotion'] ] ]
    data = data[data['cleaned_text'] != '']



    output_dir = 'kaggle/working'
    train_dataset_path = output_dir + '/train_dataset.csv'
    test_dataset_path = output_dir + '/test_dataset.csv'

    train, test = train_test_split(data, test_size=0.2,
                                    shuffle=True, random_state=42)

    train.to_csv(train_dataset_path, index=None)
    test.to_csv(test_dataset_path, index=None)

    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)




    # Defining the number of samples in train, validation and test dataset
    size_train = df_train.shape[0]
    size_test = df_test.shape[0]

    from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification

    model_path_or_name = 'kaggle/input/transformers/distilbert-base-uncased'

    # instantiate model & tokenizer
    model = AutoModel.from_pretrained(model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    seq_len = [len(i.split()) for i in df_train["cleaned_text"]]

    pd.Series(seq_len).hist(bins = range(0,128,2))

    from torch.utils.data import DataLoader, Dataset
    from transformers.data.processors.utils import InputFeatures


    # source https://github.com/hazemhosny/Emotion-Sentiment-Analysis
    class ClassificationDataset(Dataset):
        def __init__(self, text, target, model_name, max_len, label_map):
            super(ClassificationDataset).__init__()
            """
            Args:
            text (List[str]): List of the training text
            target (List[str]): List of the training labels
            tokenizer_name (str): The tokenizer name (same as model_name).
            max_len (int): Maximum sentence length
            label_map (Dict[str,int]): A dictionary that maps the class labels to integer
            """
            self.text = text
            self.target = target
            self.tokenizer_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_len = max_len
            self.label_map = label_map

        def __len__(self):
            return len(self.text)

        def __getitem__(self, item):
            text = str(self.text[item])
            text = " ".join(text.split())

            inputs = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True
            )
            return InputFeatures(**inputs, label=self.label_map[self.target[item]])


    label_map = { v:index for index, v in enumerate(list(emotions)) }
    print(label_map)
    max_len = 64
    train_dataset = ClassificationDataset(
        df_train["cleaned_text"].to_list(),
        df_train["emotion"].to_list(),
        model_path_or_name,
        max_len,
        label_map
      )
    test_dataset = ClassificationDataset(
        df_test["cleaned_text"].to_list(),
        df_test["emotion"].to_list(),
        model_path_or_name,
        max_len,
        label_map
      )

    from transformers.data.processors.utils import InputFeatures

    print(next(iter(train_dataset)))

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_path_or_name, return_dict=True, num_labels=len(label_map))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        assert len(preds) == len(p.label_ids)
        macro_f1 = f1_score(p.label_ids,preds,average='macro')
        precision = precision_score(p.label_ids,preds,average='macro')
        recall = recall_score(p.label_ids,preds,average='macro')
        acc = accuracy_score(p.label_ids,preds)
        return {
          'macro_f1' : macro_f1,
          'Accuracy': acc,
          'Precision': precision,
          'Recall' : recall,
        }


    inv_label_map = { v:k for k, v in label_map.items()}
    print(inv_label_map)


    # max_len = 64
    #
    # pred_df = pd.DataFrame([])
    # pred_df["Text"] = df_test["cleaned_text"].copy()
    # pred_df_top = pred_df.head(500)
    # max_len = 64
    #
    from transformers import pipeline
    # from tqdm import tqdm
    pipe = pipeline("sentiment-analysis", model="kaggle/working/output_dir", return_all_scores =True, max_length=max_len, truncation=True, top_k=3)
    # preds = []
    #
    # lst = ["I appreciate it, that's good to know. I hope I'll have to apply that knowledge one day", "You may die, but it's a sacrifice I'm willing to make", "Go troll elsewhere. This woman needs support, not crass questions."]
    #message = "I appreciate it, that's good to know. I hope I'll have to apply that knowledge one day"

    return pipe(message)[0][0]
    # for s in tqdm(pred_df_top["Text"].to_list()):
    #     preds.append(pipe(s)[0][0])
    #
    # pred_df_top["Prediction"] = preds
    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_rows', None)
    # print(pred_df_top)

print(emotion_classifier("fuck off"))