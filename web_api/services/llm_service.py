import os
import json
import google.generativeai as genai
from flask import Flask

KNOWLEDGE_PATH = './data/medical_knowledge.json'

class LLMService:
    def __init__(self):
        with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.medical_knowledge = json.load(f)

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_advice(self, disease_name, confidence, lang="En"):
        if disease_name not in self.medical_knowledge:
            return "Knowledge base missing for this disease." if lang == "En" else "Hệ thống chưa có thông tin chi tiết về bệnh lý này."

        knowledge = self.medical_knowledge[disease_name]

        prompt = f"""
        You are a medical virtual assistant specializing in dermatological first aid.
        The computer vision system predicted the patient has: {disease_name} with {confidence}% confidence.

        Ground Truth Medical Information:
        - Severity: {knowledge['severity']}
        - First aid: {', '.join(knowledge['first_aid'])}
        - Warning: {knowledge['warning']}

        Task:
        1. Greet the patient reassuringly and professionally.
        2. Inform them of the predicted disease and its severity level.
        3. Present the first aid steps clearly using bullet points.
        4. Emphasize the warning section.
        5. Conclude with a strict medical disclaimer: "This is a preliminary consultation based on image analysis and does not replace a professional doctor's diagnosis."
        """

        if lang == "Vi":
            prompt += "\nIMPORTANT INSTRUCTION: You MUST generate the entire final response in Vietnamese. Translate the disease name and all medical terms accurately into natural Vietnamese."
        else:
            prompt += "\nIMPORTANT INSTRUCTION: You MUST generate the entire final response in English."

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return str(e)


llm_service = LLMService()