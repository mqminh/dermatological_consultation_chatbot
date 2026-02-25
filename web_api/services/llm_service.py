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
        self.model_name = 'gemini-2.5-flash'

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

    def chat_followup(self, user_message, disease_name, history, lang="En"):
        system_instruction = f"""
        You are a medical virtual assistant specializing in dermatological first aid.
        The user was previously diagnosed with '{disease_name}' by the vision system.
        Answer their follow-up questions accurately based on this context.
        Do not prescribe medication. Always advise consulting a doctor for serious concerns.
        """

        if lang == "Vi":
            system_instruction += " IMPORTANT: You MUST answer entirely in Vietnamese."

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )

            formatted_history = []
            for msg in history:
                role = 'user' if msg['role'] == 'user' else 'model'
                text = msg.get('text', '[Image provided by user]' if role == 'user' else '')
                if text:
                    formatted_history.append({'role': role, 'parts': [text]})

            chat = model.start_chat(history=formatted_history)
            response = chat.send_message(user_message)
            return response.text
        except Exception as e:
            return str(e)

llm_service = LLMService()