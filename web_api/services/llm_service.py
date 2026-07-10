import os
import json
import urllib.request
import google.generativeai as genai

KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'medical_knowledge.json')


class LLMService:
    def __init__(self):
        with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.medical_knowledge = json.load(f)

        self.use_local_llm = os.environ.get("USE_LOCAL_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.model_name = 'gemini-2.5-flash'

    def _should_use_local(self, llm_mode=None):
        if llm_mode is not None:
            mode = str(llm_mode).strip().lower()
            if mode in {"local", "ollama", "local_llm", "true"}:
                return True
            if mode in {"gemini", "remote", "false", ""}:
                return False

        return self.use_local_llm

    def _call_ollama(self, prompt):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }

        request = urllib.request.Request(
            f"{self.ollama_host.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
            return result.get("response", "").strip()

    def generate_advice(self, disease_name, confidence, lang="En", llm_mode=None):
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

        if self._should_use_local(llm_mode):
            try:
                return self._call_ollama(prompt)
            except Exception:
                pass

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return str(e)

    def chat_followup(self, user_message, disease_name, history, lang="En", llm_mode=None):
        system_instruction = f"""
        You are a medical virtual assistant specializing in dermatological first aid.
        The user was previously diagnosed with '{disease_name}' by the vision system.
        Answer their follow-up questions accurately based on this context.
        Do not prescribe medication. Always advise consulting a doctor for serious concerns.
        """

        if lang == "Vi":
            system_instruction += " IMPORTANT: You MUST answer entirely in Vietnamese."

        try:
            prompt = f"{system_instruction}\n\nConversation history:\n"
            for msg in history:
                role = 'User' if msg['role'] == 'user' else 'Assistant'
                text = msg.get('text', '')
                if text:
                    prompt += f"{role}: {text}\n"

            prompt += f"\nUser now asks: {user_message}"

            if self._should_use_local(llm_mode):
                try:
                    return self._call_ollama(prompt)
                except Exception:
                    pass

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return str(e)


llm_service = LLMService()