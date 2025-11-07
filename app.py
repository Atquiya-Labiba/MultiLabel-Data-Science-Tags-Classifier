from flask import Flask, render_template, request
from gradio_client import Client

app = Flask(__name__)
HF_SPACE = "atquiyaoni/Multilabel-DataScience-Tags-Classifier"
client = Client(HF_SPACE)

def predict_tags(input_text):
    result = client.predict(
        description=input_text,
        api_name="/predict"
    )
    if not result:        
        return {"confidences": [{"label": "No tags returned", "confidence": 1.0}]}
    return result

@app.route("/", methods=['GET', 'POST'])
def index():
    input_text = ""
    output_text = ""

    if request.method == "POST":
        input_text = request.form.get('text', '').strip()
        if not input_text:
            output_text = "Please enter a question before submitting."
            return render_template("index.html", input_text=input_text, output_text=output_text)
        
        result = predict_tags(input_text)
        
        confidence_list = result.get("confidences", [])
        labels = [item["label"] for item in confidence_list if item.get("confidence", 0) ]
        
        output_text = ", ".join(labels)       
        return render_template("result.html", input_text=input_text, output_text=output_text)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
