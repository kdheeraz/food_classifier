from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CLASS_LABELS = ['adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch', 'aloo_tikki', 'anarsa', 'ariselu', 'bandar_laddu', 'basundi', 'bhatura', 'bhindi_masala', 'biryani', 'boondi', 'butter_chicken', 'chak_hao_kheer', 'cham_cham', 'chana_masala', 'chapati', 'chhena_kheeri', 'chicken_razala', 'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'daal_baati_churma', 'daal_puri', 'dal_makhani', 'dal_tadka', 'dharwad_pedha', 'doodhpak', 'double_ka_meetha', 'dum_aloo', 'gajar_ka_halwa', 'gavvalu', 'ghevar', 'gulab_jamun', 'imarti', 'jalebi', 'kachori', 'kadai_paneer', 'kadhi_pakoda', 'kajjikaya', 'kakinada_khaja', 'kalakand', 'karela_bharta', 'kofta', 'kuzhi_paniyaram', 'lassi', 'ledikeni', 'litti_chokha', 'lyangcha', 'maach_jhol', 'makki_di_roti_sarson_da_saag', 'malapua', 'misi_roti', 'misti_doi', 'modak', 'mysore_pak', 'naan', 'navrattan_korma', 'palak_paneer', 'paneer_butter_masala', 'phirni', 'pithe', 'poha', 'poornalu', 'pootharekulu', 'qubani_ka_meetha', 'rabri', 'ras_malai', 'rasgulla', 'sandesh', 'shankarpali', 'sheer_korma', 'sheera', 'shrikhand', 'sohan_halwa', 'sohan_papdi', 'sutar_feni', 'unni_appam']
# Load your custom model
model = load_model('india_food.h5')

def preprocess_image(img):
    img = img.resize((250,200))  # ImageNet models typically use 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess input as required for ImageNet
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        
        if predictions.shape[1] == 1000:
            # Use ImageNet decoding
            decoded_predictions = decode_predictions(predictions, top=1)[0][0]
            predicted_label = decoded_predictions[1]
            confidence = float(decoded_predictions[2])
        else:
            # For custom datasets, you should use a custom class mapping
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            predicted_label = f'Class {CLASS_LABELS[predicted_index]}'  # Replace with your custom class labels if available

        result = {
            'label': predicted_label,
            'confidence': confidence
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
