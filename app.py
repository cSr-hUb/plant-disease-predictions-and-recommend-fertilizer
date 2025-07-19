from flask import Flask, render_template, request, url_for, send_from_directory
from collections import Counter
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
from g4f.client import Client

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the GPT client
client = Client()

# System prompt for fertilizer suggestions
SYSTEM_PROMPT = """
You are an expert agronomist specializing in rice plant health. Your task is to provide precise and practical fertilizer suggestions based on the detected rice plant diseases. For each disease provided, recommend specific fertilizers (e.g., nitrogen-based, potassium-rich, etc.), application methods (e.g., foliar spray, soil application), and any additional care tips to mitigate the disease. Ensure your suggestions are concise, actionable, and tailored to rice crops. If multiple diseases are detected, prioritize recommendations that address all conditions effectively.
"""

def boundingboxPredicted(results, model, image_path):
    output_folder = 'predictions'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)

    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes  # Corrected from "r Boxes"
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        output_image_path = os.path.join(output_folder, 'predictions.jpg')
        cv2.imwrite(output_image_path, img)
        print(f"Predictions saved in {output_folder}")
        return output_image_path

def run_object_detection(image_path):
    model_directory = r"C:\Users\sri\Desktop\plant disease\RICE PLANT YOLO-28 CLASS\website"
    model_filename = "best.pt"
    model_path = os.path.join(model_directory, model_filename)

    infer = YOLO(model_path)
    result = infer.predict(image_path)
    item_counts = Counter(infer.names[int(c)] for r in result for c in r.boxes.cls)
    object_list = list(item_counts.keys())

    return object_list, result, infer

def get_fertilizer_suggestions(diseases):
    user_prompt = f"Detected rice plant diseases: {', '.join(diseases)}. Provide fertilizer suggestions and care tips."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        web_search=False
    )
    
    return response.choices[0].message.content

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('result.html', error="No file part")

    image_file = request.files['image']

    if image_file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('result.html', error="Invalid file type")

    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    
    print(f"Image saved at: {image_path}")
    
    object_list, result, infer = run_object_detection(image_path)
    predicted_image_path = boundingboxPredicted(result, infer, image_path)

    fertilizer_suggestions = get_fertilizer_suggestions(object_list)

    return render_template('result.html', 
                         image_filename=image_file.filename, 
                         object_list=object_list, 
                         predicted_image_path=predicted_image_path,
                         fertilizer_suggestions=fertilizer_suggestions)

@app.route('/static/predictions/<filename>')
def predictions(filename):
    return send_from_directory('predictions', filename)

if __name__ == '__main__':
    app.run(debug=True)