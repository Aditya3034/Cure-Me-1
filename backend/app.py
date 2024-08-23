from flask import Flask, jsonify,request
from flask_cors import CORS
import modules.RAG.index as qa

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Healthy"}), 200

@app.route('/qaretrival', methods=['POST'])
def qaretrival():
    # Replace this with your logic for QA retrieval
    data = request.json
    if 'question' in data:
        try:
            question = data['question']
            r = qa.predict(question)
            result = r['result']
            return jsonify({"Success":True,"Message":result})
        except Exception as e:
            return jsonify({"error":f"An error occured {e}"}),500

if __name__ == '__main__':
    app.run(debug=True)
