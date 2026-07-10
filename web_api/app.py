import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from controllers.prediction_controller import prediction_bp
from controllers.consultation_controller import consultation_bp

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.register_blueprint(prediction_bp, url_prefix='/api')
app.register_blueprint(consultation_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)