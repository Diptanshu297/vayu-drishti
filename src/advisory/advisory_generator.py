"""
Generate multilingual health advisories using Claude API.
Falls back to templates when API is unavailable.
"""

import json
import os

import anthropic

from config.settings import AQI_CATEGORIES


def generate_advisory(
    forecast_data: dict,
    attribution_data: dict | None,
    audience: str,
    language: str,
    ward_name: str,
    city: str = "Kolkata",
) -> dict:
    """
    Generate a health advisory using Claude, with template fallback.

    Args:
        forecast_data: dict with keys like aqi, dominant_pollutant, duration_hours
        attribution_data: dict from compute_source_attribution, or None
        audience: one of "school", "administrator", "citizen", "worker"
        language: one of "bengali", "hindi", "kannada", "english"
        ward_name: name of the ward/area
        city: city name

    Returns:
        dict with:
            - advisory: the generated text
            - source: "llm" or "template"
            - language: language used
            - audience: audience type
            - verification: back-translation (if LLM)
    """
    try:
        return _generate_llm_advisory(
            forecast_data, attribution_data, audience, language, ward_name, city
        )
    except Exception as e:
        print(f"LLM advisory failed ({e}), falling back to template")
        return _generate_template_advisory(
            forecast_data, audience, language, ward_name
        )


def _generate_llm_advisory(
    forecast_data: dict,
    attribution_data: dict | None,
    audience: str,
    language: str,
    ward_name: str,
    city: str,
) -> dict:
    """Generate advisory via Claude API."""
    client = anthropic.Anthropic()

    language_names = {
        "bengali": "Bengali (বাংলা)",
        "hindi": "Hindi (हिन्दी)",
        "kannada": "Kannada (ಕನ್ನಡ)",
        "english": "English",
    }

    system_prompt = f"""You are a public health communication specialist for the {city} Municipal Corporation's air quality advisory system.

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
- At the end, add a section marked [VERIFICATION] with an English back-translation of the advisory above. This is for quality checking and will be stripped before display.
"""

    user_message = f"""Generate a health advisory based on this data:

Forecast: {json.dumps(forecast_data, ensure_ascii=False)}
{"Attribution: " + json.dumps(attribution_data, ensure_ascii=False) if attribution_data else "Attribution: not available"}

Generate an immediately actionable advisory for {audience} in {ward_name}, {city}.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    full_text = response.content[0].text

    # Split advisory from verification
    if "[VERIFICATION]" in full_text:
        parts = full_text.split("[VERIFICATION]")
        advisory = parts[0].strip()
        verification = parts[1].strip() if len(parts) > 1 else ""
    else:
        advisory = full_text.strip()
        verification = ""

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
    aqi = forecast_data.get("aqi", 0)
    category = _get_category_label(aqi)
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

    # Map AQI to severity key
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


def _get_category_label(aqi: int) -> str:
    for cat in AQI_CATEGORIES:
        low, high = cat["range"]
        if low <= aqi <= high:
            return cat["label"]
    return "Severe"
