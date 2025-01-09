import re

# Given string
given_string = "\
                The image depicts a joyful group of athletes celebrating while holding a trophy, with excitement evident on their faces. They are dressed in blue and white uniforms with medals around their necks.\
            \
            Text present in the image:\
            - 11\
            - AFA\
            - FIFA WORLD CUP QATAR 2022 (on the medals)\
            \
            Dominant Emotion:\
            - Happiness: 90%\
            \
            Other Emotions:\
            - Excitement: 70%\
            - Pride: 60%\
            - Relief: 40%\
        "

# Extracting description
description = given_string.split("Text present in the image:")[0].strip()

# Extracting text present in the image
# text_in_image = re.findall(r'-\s*([^\n]+)', given_string.split("Text present in the image:")[1].strip().split('Dominant Emotion')[0].strip())
text_in_image = ', '.join(re.findall(r'-\s*([^\n]+)', given_string.split("Text present in the image:")[1].strip().split('Dominant Emotion')[0]))
# Extracting dominant emotion
dominant_emotion = re.findall(r'-\s*([^\n:]+):\s*(\d+%)', given_string.split("Dominant Emotion:")[1])[0]

# Extracting other emotions
other_emotions = dict(re.findall(r'-\s*([^\n:]+):\s*(\d+%)', given_string.split("Other Emotions:")[1]))

# Printing the parsed data
print("Description:", description)
print("Text present in the image:", text_in_image)
print("Dominant Emotion:", dominant_emotion[0])
print("Other Emotions:")
for emotion, percentages in other_emotions.keys()[:2]:
    print( emotion, percentages)
