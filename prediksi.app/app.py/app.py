from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Muat model yang telah dilatih
with open('path_to_your_model/car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Ambil data dari permintaan
    brand = data.get('merk')
    year = data.get('tahun')
    mileage = data.get('kilometer')
    transmission = data.get('transmisi')
    fuel = data.get('bahan_bakar')
    engine_size = data.get('cc_mesin')
    condition = data.get('kondisi')

    # Buat DataFrame untuk input
    input_data = pd.DataFrame({
        'brand': [brand],
        'year': [year],
        'mileage': [mileage],
        'transmission': [transmission],
        'fuel': [fuel],
        'engine_size': [engine_size],
        'condition': [condition]
    })

    # Lakukan prediksi
    prediction = model.predict(input_data)

    # Kembalikan hasil prediksi
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
