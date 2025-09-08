from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2  # For image reading
import numpy as np
import tensorflow as tf  # For loading the model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load the trained CNN model
model = tf.keras.models.load_model('skin_model.h5')

# ✅ Labels matching model training
CLASS_LABELS = ['acne', 'dry_skin', 'oily_skin']

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"[INFO] Image saved at: {filepath}")

    # ✅ Load and preprocess image to match training size (224x224)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 400

    try:
        img = cv2.resize(image, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)

        prediction = model.predict(img)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # ✅ Recommendations based on prediction
    recommendations = {
        "acne": ["Neem Face Wash", "Aloe Vera Gel", "Turmeric Paste"],
        "dry_skin": ["Coconut Oil", "Shea Butter", "Hydrating Cream"],
        "oily_skin": ["Multani Mitti", "Rose Water", "Oil-Free Moisturizer"]
    }

    return jsonify({
        'condition': predicted_class,
        'recommended_products': recommendations.get(predicted_class, [])
    })


if __name__ == '__main__':
    app.run(debug=True)


