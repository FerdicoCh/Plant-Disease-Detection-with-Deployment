import os
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model-resnet50-best.keras'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)
class_names = sorted(os.listdir(r"C:\Users\USER\Desktop\Get back to Work\Plant Disese Data\PlantVillage"))

# Deskripsi penyakit
disease_info = {
    "Pepper__bell___Bacterial_spot": "Penyakit bercak bakteri pada paprika, ditandai dengan bintik hitam basah pada daun dan buah.",
    "Pepper__bell___healthy": "Tanaman paprika dalam kondisi sehat.",
    "Potato___Early_blight": "Penyakit jamur yang menimbulkan bercak gelap melingkar pada daun kentang.",
    "Potato___Late_blight": "Penyakit jamur mematikan pada kentang yang menyebabkan daun menghitam dan busuk.",
    "Potato___healthy": "Tanaman kentang dalam kondisi sehat.",
    "Tomato_Bacterial_spot": "Penyakit bakteri pada tomat yang menyebabkan bintik kecil pada daun dan buah.",
    "Tomato_Early_blight": "Jamur Alternaria menyebabkan bercak melingkar dengan lingkaran konsentris pada daun.",
    "Tomato_Late_blight": "Penyakit serius yang membuat daun tomat hitam dan tanaman cepat layu.",
    "Tomato_Leaf_Mold": "Jamur pada tomat yang berkembang di bawah daun dengan lapisan berwarna kuning atau coklat.",
    "Tomato_Septoria_leaf_spot": "Penyakit dengan bercak kecil berbatasan gelap dan bagian tengah terang.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Serangan tungau kecil yang meninggalkan bintik kuning dan jaring pada daun.",
    "Tomato__Target_Spot": "Ditandai dengan bercak berwarna gelap yang berkembang menjadi cincin konsentris pada daun tomat.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Virus yang menyebabkan daun tomat menggulung, menguning, dan pertumbuhan kerdil.",
    "Tomato__Tomato_mosaic_virus": "Infeksi virus yang menyebabkan daun tomat keriting, warna belang dan pertumbuhan terhambat.",
    "Tomato_healthy": "Tanaman tomat dalam kondisi sehat."
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return render_template('index.html', prediction="Tidak ada file.")

    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    img = image.load_img(save_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_name = class_names[class_index]
    class_name_clean = class_name.replace("___", " ").replace("_", " ")
    explanation = disease_info.get(class_name, "Deskripsi belum tersedia.")

    image_url = url_for('static', filename=f'uploads/{filename}')
    return render_template('index.html',
                           prediction=class_name_clean,
                           image_path=image_url,
                           explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
