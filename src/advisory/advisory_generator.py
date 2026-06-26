"""
Generate multilingual health advisories using Google Gemini API (free).
Falls back to templates when API is unavailable.
"""

import json
import os

import requests

from config.settings import AQI_CATEGORIES


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

def generate_advisory(
    forecast_data: dict,
    attribution_data: dict | None,
    audience: str,
    language: str,
    ward_name: str,
    city: str = "Kolkata",
) -> dict:
    """
    Generate a health advisory using Gemini, with template fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _generate_template_advisory(forecast_data, audience, language, ward_name)

    try:
        return _generate_llm_advisory(
            forecast_data, attribution_data, audience, language, ward_name, city, api_key
        )
    except Exception as e:
        print(f"LLM advisory failed ({e}), falling back to template")
        return _generate_template_advisory(forecast_data, audience, language, ward_name)


def _generate_llm_advisory(
    forecast_data: dict,
    attribution_data: dict | None,
    audience: str,
    language: str,
    ward_name: str,
    city: str,
    api_key: str,
) -> dict:
    """Generate advisory via Google Gemini API."""
    language_names = {
        "bengali": "Bengali (বাংলা)",
        "hindi": "Hindi (हिन्दी)",
        "kannada": "Kannada (ಕನ್ನಡ)",
        "english": "English",
    }

    prompt = f"""You are a public health communication specialist for the {city} Municipal Corporation's air quality advisory system.

Your audience: {audience}
Language: {language_names.get(language, language)}
Ward/Area: {ward_name}, {city}

Rules:
- Write the advisory ENTIRELY in {language_names.get(language, language)}. No English words unless universally understood (AQI, PM2.5).
- For schools: focus on child safety actions. No technical jargon. Keep it under 4 sentences.
- For administrators: include source attribution data, relevant regulatory references (Air Act 1981, CPCB/SPCB guidelines), and specific enforcement actions. Be detailed.
- For citizens/workers: keep it under 3 sentences. Actionable advice only. Use the AQI category name, not raw numbers.
- Never use alarming language. Be direct, calm, factual.
- Include time window ("until tomorrow evening", not "for 18 hours").

Generate a health advisory based on this data:

Forecast: {json.dumps(forecast_data, ensure_ascii=False)}
{"Attribution: " + json.dumps(attribution_data, ensure_ascii=False) if attribution_data else "Attribution: not available"}

Generate an immediately actionable advisory for {audience} in {ward_name}, {city}.

After the advisory, add a section starting with [VERIFICATION] containing an English back-translation for quality checking.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 800,
        },
    }

    response = requests.post(
        f"{GEMINI_URL}?key={api_key}",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    full_text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Split advisory from verification
    if "[VERIFICATION]" in full_text:
        parts = full_text.split("[VERIFICATION]")
        advisory = parts[0].strip()
        verification = parts[1].strip() if len(parts) > 1 else ""
    else:
        advisory = full_text.strip()
        verification = ""

    # Clean markdown artifacts
    advisory = advisory.replace("**", "").replace("##", "").replace("# ", "")

    return {
        "advisory": advisory,
        "source": "llm",
        "language": language,
        "audience": audience,
        "verification": verification,
    }


def _generate_template_advisory(
    forecast_data: dict,
    audience: str,
    language: str,
    ward_name: str,
) -> dict:
    """Template-based fallback when LLM is unavailable."""
    aqi = forecast_data.get("aqi", 0) or 0
    hours = forecast_data.get("duration_hours", 24)

    templates = {
        ("citizen", "bengali"): {
            "severe": f"সতর্কতা: {ward_name} এলাকায় আগামী {hours} ঘণ্টা বায়ুর মান অত্যন্ত খারাপ থাকবে (AQI {aqi})। বাইরে যাওয়া এড়িয়ে চলুন। মাস্ক ব্যবহার করুন।",
            "poor": f"সতর্কতা: {ward_name} এলাকায় বায়ুর মান খারাপ (AQI {aqi})। সংবেদনশীল ব্যক্তিরা বাইরের কার্যকলাপ সীমিত করুন।",
            "moderate": f"{ward_name} এলাকায় বায়ুর মান মোটামুটি। স্বাভাবিক কার্যকলাপ চালিয়ে যেতে পারেন।",
            "good": f"{ward_name} এলাকায় বায়ুর মান ভালো আছে।",
        },
        ("citizen", "hindi"): {
            "severe": f"चेतावनी: {ward_name} में अगले {hours} घंटे हवा बहुत खराब रहेगी (AQI {aqi})। बाहर जाने से बचें। मास्क पहनें।",
            "poor": f"चेतावनी: {ward_name} में हवा की गुणवत्ता खराब है (AQI {aqi})। संवेदनशील लोग बाहरी गतिविधियाँ सीमित करें।",
            "moderate": f"{ward_name} में हवा की गुणवत्ता ठीक है। सामान्य गतिविधियाँ जारी रख सकते हैं।",
            "good": f"{ward_name} में हवा की गुणवत्ता अच्छी है।",
        },
        ("citizen", "kannada"): {
            "severe": f"ಎಚ್ಚರಿಕೆ: {ward_name} ಪ್ರದೇಶದಲ್ಲಿ ಮುಂದಿನ {hours} ಗಂಟೆಗಳ ಕಾಲ ಗಾಳಿಯ ಗುಣಮಟ್ಟ ತೀವ್ರವಾಗಿ ಕೆಟ್ಟದಾಗಿರುತ್ತದೆ (AQI {aqi})। ಹೊರಗೆ ಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಿ। ಮಾಸ್ಕ್ ಧರಿಸಿ.",
            "poor": f"ಎಚ್ಚರಿಕೆ: {ward_name} ಪ್ರದೇಶದಲ್ಲಿ ಗಾಳಿಯ ಗುಣಮಟ್ಟ ಕಳಪೆಯಾಗಿದೆ (AQI {aqi})। ಸೂಕ್ಷ್ಮ ವ್ಯಕ್ತಿಗಳು ಹೊರಾಂಗಣ ಚಟುವಟಿಕೆಗಳನ್ನು ಮಿತಿಗೊಳಿಸಿ.",
            "moderate": f"{ward_name} ಪ್ರದೇಶದಲ್ಲಿ ಗಾಳಿಯ ಗುಣಮಟ್ಟ ಮಧ್ಯಮವಾಗಿದೆ. ಸಾಮಾನ್ಯ ಚಟುವಟಿಕೆಗಳನ್ನು ಮುಂದುವರಿಸಬಹುದು.",
            "good": f"{ward_name} ಪ್ರದೇಶದಲ್ಲಿ ಗಾಳಿಯ ಗುಣಮಟ್ಟ ಉತ್ತಮವಾಗಿದೆ.",
        },
    }

    if aqi > 300:
        severity = "severe"
    elif aqi > 200:
        severity = "poor"
    elif aqi > 100:
        severity = "moderate"
    else:
        severity = "good"

    key = (audience if audience == "citizen" else "citizen", language)
    template_set = templates.get(key, templates.get(("citizen", "hindi"), {}))
    advisory_text = template_set.get(severity, f"AQI: {aqi} in {ward_name}.")

    return {
        "advisory": advisory_text,
        "source": "template",
        "language": language,
        "audience": audience,
        "verification": "",
    }