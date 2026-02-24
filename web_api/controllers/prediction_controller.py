import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.prediction_service import prediction_service

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict', methods=['POST'])
def predict_skin_disease():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Missing file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = prediction_service.predict(file_path)

        os.remove(file_path)

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500