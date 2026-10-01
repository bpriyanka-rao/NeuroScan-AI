import os

# Prefer the newer google.genai SDK when available.
try:
    import google.genai as genai
    GENAI_SDK = "genai"
except ImportError:
    try:
        import google.generativeai as genai
        GENAI_SDK = "generativeai"
    except ImportError:
        genai = None
        GENAI_SDK = None


def _format_chat_error(error: Exception) -> str:
    message = str(error)
    if "api_key_invalid" in message.lower() or "api key not valid" in message.lower():
        return "Gemini rejected the API key. Set a valid GEMINI_API_KEY in .env and restart the app."
    return f"Sorry, I encountered an error: {message}"


def get_chatbot_response(user_message: str, current_scan_context: str = None) -> str:
    """
    Sends a message to the Gemini chatbot and returns the response.
    """
    if genai is None:
        return (
            "Chatbot unavailable. Install the latest Gemini SDK and set GEMINI_API_KEY. "
            "See README for setup details."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the environment."

    if GENAI_SDK == "genai":
        client = genai.LanguageServiceClient()
        system_prompt = (
            "You are NeuroScan AI's medical assistant. You help users understand "
            "Alzheimer's disease, MRI scan results, Grad-CAM heatmaps, and the "
            "4 stages: Non Demented, Very Mild, Mild, Moderate. Be clear, "
            "compassionate, and medically accurate. Never give direct medical advice."
        )
        if current_scan_context:
            system_prompt += f"\n\n{current_scan_context}"

        try:
            response = client.generate_text(
                model="gemini-1.5-flash",
                prompt=f"{system_prompt}\n\nUser: {user_message}",
                temperature=0.7,
            )
            return response.text
        except Exception as e:
            return _format_chat_error(e)
    else:
        genai.configure(api_key=api_key)
        system_prompt = (
            "You are NeuroScan AI's medical assistant. You help users understand "
            "Alzheimer's disease, MRI scan results, Grad-CAM heatmaps, and the "
            "4 stages: Non Demented, Very Mild, Mild, Moderate. Be clear, "
            "compassionate, and medically accurate. Never give direct medical advice."
        )
        if current_scan_context:
            system_prompt += f"\n\n{current_scan_context}"

        try:
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_message)
            return response.text
        except Exception as e:
            return _format_chat_error(e)
