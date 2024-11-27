from flask import Flask, request, jsonify
from flask_cors import CORS
import markdown
from model3 import get_answer  # Importing the get_answer function from model2.py

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

def format_response(text):
    """
    Convert the response text into HTML format using markdown.
    """
    formatted_text = f"{text}"
    html = markdown.markdown(formatted_text)
    return html

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles POST requests to the /chat endpoint.
    Expects a JSON body with a 'message' field.
    """
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Get the answer from the LangChain model
        response = get_answer(user_input)
        formatted_response = format_response(response)  # Format the answer into HTML
        return jsonify({"response": formatted_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode for better error tracking
