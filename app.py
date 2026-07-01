import os
import json
import uuid
import re
import statistics
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv
import math

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

LOG_FILE = "audit_log.json"

# init in-memory rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            json.dump([], f)

def append_to_log(entry):
    with open(LOG_FILE, 'r+') as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=4)

def get_log():
    with open(LOG_FILE, 'r') as f:
        return json.load(f)

def analyze_with_llm(text):
    # signal 1 measures semantic patterns and uniform tone.
    prompt = """
    Analyze the text for AI generation markers (semantic predictability, structural monotony).
    Look aggressively for:
    1. Corporate or academic transition words ("Furthermore", "It is important to note", "Additionally").
    2. Symmetrical sentence structures and neutral, objective tones.
    3. Lack of colloquialisms, personal bias, or erratic punctuation.
    4. Excessive presence of unusual punctuation or formatting (e.g., excessive use of semicolons, colons, em dashes).
    If these markers exist, score heavily towards 1.0.
    Output strictly valid JSON with a single key "ai_probability" mapping to a float between 0.0 (human) and 1.0 (AI).
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return float(result.get("ai_probability", 0.0))

def analyze_with_heuristics(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.lower().split()
    
    if len(words) < 3:
        return 0.5
        
    # Metric 1: Word Length Variance
    # Computes standard deviation of characters per word.
    word_lengths = [len(w) for w in words]
    word_var = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
    wl_score = max(0.0, 1.0 - (word_var / 3.0)) 
    
    # Metric 2: Sentence Length Variance
    sent_lengths = [len(s.split()) for s in sentences]
    sent_var = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
    sl_score = max(0.0, 1.0 - (sent_var / 15.0))
    
    # Metric 3: Herdan's C (Logarithmic TTR)
    # Normalizes vocabulary diversity against string length constraints.
    unique_words = len(set(words))
    herdans_c = math.log(unique_words) / math.log(len(words))
    ttr_score = max(0.0, 1.0 - herdans_c)
    
    # Dynamic Aggregation
    # Shifts weight away from sentence variance if the array lacks data points.
    if len(sentences) < 3:
        return (wl_score * 0.6) + (ttr_score * 0.4)
    
    return (sl_score * 0.4) + (wl_score * 0.3) + (ttr_score * 0.3)

def calculate_confidence(llm_score, heuristic_score):
    # weight the semantic evaluation of the llm heavier than the heuristics
    final_ai_score = (llm_score * 0.65) + (heuristic_score * 0.35)
    
    if final_ai_score <= 0.35:
        attribution = "human"
        confidence = 1.0 - (final_ai_score / 0.35) 
    elif final_ai_score <= 0.65:
        attribution = "uncertain"
        confidence = 1.0 - (abs(final_ai_score - 0.5) / 0.15) 
    else:
        attribution = "ai"
        confidence = (final_ai_score - 0.65) / 0.35 
        
    return final_ai_score, attribution, round(confidence, 4)

@app.route('/submit', methods=['POST'])
@limiter.limit("10 per minute; 100 per day")
def submit():
    data = request.get_json()
    text = data.get('text', '')
    creator_id = data.get('creator_id', '')

    if not text or not creator_id:
        return jsonify({"error": "Missing text or creator_id"}), 400

    content_id = str(uuid.uuid4())
    
    try:
        llm_score = analyze_with_llm(text)
        heuristic_score = analyze_with_heuristics(text)
        final_ai_score, attribution, confidence = calculate_confidence(llm_score, heuristic_score)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    log_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attribution": attribution,
        "confidence": confidence,
        "final_ai_score": round(final_ai_score, 4),
        "llm_score": round(llm_score, 4),
        "heuristic_score": round(heuristic_score, 4),
        "status": "classified" 
    }
    
    print(f"DEBUG | LLM: {llm_score} | Heuristic: {heuristic_score} | Final AI: {final_ai_score}")
    append_to_log(log_entry)

    display_conf = round(confidence * 100, 1)
    
    if attribution == "human":
        label_text = f"Written by a Human (Confidence: {display_conf}%) — This content matches normal human writing patterns, showing natural stylistic and structural variety."
    elif attribution == "uncertain":
        label_text = "Uncertain — The system detected mixed signals. This text contains a blend of highly uniform structures and personal stylistic variations. Reader discretion advised."
    else:
        label_text = f"Generated by an AI (Confidence: {display_conf}%) — This text exhibits heavy statistical markers, structural uniformity, and phrasing patterns typical of automated language models."

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": label_text
    })

@app.route('/appeal', methods=['POST'])
def appeal():
    data = request.get_json()
    target_id = data.get('content_id', '')
    reasoning = data.get('creator_reasoning', '')

    if not target_id or not reasoning:
        return jsonify({"error": "Missing content_id or creator_reasoning"}), 400

    logs = get_log()
    record_updated = False

    for entry in logs:
        if entry.get("content_id") == target_id:
            entry["status"] = "under_review" 
            entry["creator_reasoning"] = reasoning 
            record_updated = True
            break

    if not record_updated:
        return jsonify({"error": "Record not found."}), 404

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

    return jsonify({
        "status": "under_review", 
        "message": "Appeal logged successfully."
    })

@app.route('/log', methods=['GET'])
def fetch_log():
    return jsonify({"entries": get_log()})

if __name__ == '__main__':
    init_log()
    app.run(debug=True)