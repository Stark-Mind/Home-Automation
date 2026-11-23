from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Maintains a rolling history of the last 5 actions for context traps
action_history = []

@app.route('/aegis', methods=['POST'])
def aegis_endpoint():
    data = request.json
    telemetry = json.dumps(data.get("telemetry", {}))
    user_input = data.get("input", "")
    
    # Format the strict prompt exactly as trained
    prompt = f"Telemetry: {telemetry}\nHistory: {json.dumps(action_history)}\nInput: {user_input}"
    
    payload = {
        "model": "aegis",
        "prompt": prompt,
        "stream": False
    }
    
    # Send to local Ollama
    response = requests.post(OLLAMA_URL, json=payload)
    ai_command = response.json().get("response", "").strip()
    
    # Parse the response to update history
    try:
        parsed_cmd = json.loads(ai_command)
        if parsed_cmd.get("device") != "none":
            action_history.append(f"{parsed_cmd.get('device')} -> {parsed_cmd.get('state')}")
            if len(action_history) > 5:
                action_history.pop(0)
    except json.JSONDecodeError:
        pass
        
    # Send the raw JSON string back to the ESP32
    return jsonify({"command": ai_command})

if __name__ == '__main__':
    # host='0.0.0.0' exposes the server to your local Wi-Fi network
    app.run(host='0.0.0.0', port=5000)