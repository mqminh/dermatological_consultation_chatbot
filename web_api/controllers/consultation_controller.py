import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.prediction_service import prediction_service
from services.llm_service import llm_service

consultation_bp = Blueprint('consultation', __name__)


@consultation_bp.route('/consult', methods=['POST'])
def consult():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Missing file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    lang = request.form.get('lang', 'En')

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        pred_result = prediction_service.predict(file_path)
        disease = pred_result['disease']
        confidence = pred_result['confidence']

        advice = llm_service.generate_advice(disease, confidence, lang)

        os.remove(file_path)

        return jsonify({
            "success": True,
            "data": {
                "disease": disease,
                "confidence": confidence,
                "language": lang,
                "consultation": advice
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@consultation_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"success": False, "message": "Missing message"}), 400

    user_message = data['message']
    disease = data.get('disease', 'Unknown')
    history = data.get('history', [])
    lang = data.get('lang', 'En')

    try:
        reply = llm_service.chat_followup(user_message, disease, history, lang)
        return jsonify({
            "success": True,
            "data": {
                "reply": reply
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500