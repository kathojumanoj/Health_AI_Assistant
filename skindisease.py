from flask import Flask, render_template,request,jsonify
from keras.models import load_model
import google.generativeai as genai
from PIL import Image, ImageOps
from dotenv import load_dotenv
import numpy as np
import base64
import io
import os

app = Flask(__name__)

load_dotenv()
os.environ['GOOGLE_API_KEY']="your api key"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

#Model :
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
modeldl = load_model(r"D:\AIResumeProject\templates\keras_model.h5", compile=False)
# Load the labels
class_names = open(r"D:\AIResumeProject\templates\labels.txt", "r").readlines()


# Create the array of the right shape to feed into the keras model
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
@app.route('/',methods=['GET','POST'])
def home():
  return render_template('./index.html')
@app.route('/send_message', methods=['POST','GET'])
def send_message():
    user_message = request.json['user_message']
    print(user_message)
    # user_message=user_message
    # t= user_message+"is this question related to Medical field give answer (0 or 1)"
    # outz=get_gemini_response(t)
    # print()
    # print()
    # print("response :",outz)
    # print()
    # if outz==1 or 'Yes' in outz:
    response_message=get_gemini_response(user_message)
    response_message='<pre>'+response_message+'</pre>'
    # else:
    #     response_message="Plz ask me medical related quries ... :)"

    return jsonify({'response_message': response_message})

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image file from the request
    image_file = request.files['image']

    if image_file:
        # Read the image file as bytes
        image_data = image_file.read()

        # Convert the image data to a base64 string
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image_bytes = io.BytesIO(base64.b64decode(image_base64))

        # Open the image using PIL (Pillow)
        image = Image.open(image_bytes)


        # Display the image
        # display.display(image)
        # Resize and preprocess the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        image_data = normalized_image_array

# image_data = load_and_preprocess_image(image_path)

        if image_data is not None:
            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = image_data

            # Predict the model
            prediction = modeldl.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            ans=class_name[2:]
            # Print prediction and confidence score
            print("Confidence Score:", confidence_score)
            print("Class:", class_name[2:])
            print()
            print()
            print("Answer :",ans)
            print()
            user_message="i have "+ans+" disease and give me reasons and solutions? Give Reasons and Solutions in Steps."
            response_message=get_gemini_response(user_message)
            response_message=ans+response_message
            return jsonify({'response_message': response_message})

    else:
        return jsonify({'result': 'No image uploaded'})


app.run(debug=True)
