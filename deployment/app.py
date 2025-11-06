import gradio as gr
import onnxruntime as rt
from transformers import AutoTokenizer
import torch, json

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")


with open("tag_types_encoded.json", "r") as fp:
  encode_tag_types = json.load(fp)

tags = list(encode_tag_types.keys())

inf_session = rt.InferenceSession('question-classifier-quantized.onnx')
input_name = inf_session.get_inputs()[0].name
output_name = inf_session.get_outputs()[0].name

def classify_question_tags(description):
  input_ids = tokenizer(description)['input_ids'][:512]
  logits = inf_session.run([output_name], {input_name: [input_ids]})[0]
  logits = torch.FloatTensor(logits)
  probs = torch.sigmoid(logits)[0]  
  
  return dict(zip(tags, map(float, probs))) 

demo = gr.Interface(
    fn=classify_question_tags,
    inputs=gr.Textbox(lines=8, placeholder="Enter your question here..."),
    outputs=gr.Label(num_top_classes=5), 
    examples = [    
    "I want to develop a machine learning model that predicts the correct medicine dosage required to keep a specific lab value within the target range of 5 to 7. I also have several other predictor variables available. I am unsure which machine learning algorithm would be most suitable for deployment and use with future patients. Additionally, should I define the outcome as binary (1 if the value is between 5 and 7, and 0 otherwise), or is there a better approach?",
    "What is the best way to evaluate performance of Generative Adverserial Network (GAN)? Perhaps measuring the distance between two distributions or maybe something else?",
    "Suppose that I have a file which has thousands of skills starting from A-Z. Now, I would like to create a model that can group similar skills together (example neural network and SVM can group together). I know that I can use NLP for this problem, but I'm not sure about the algorithm that I can use to get the best result."
],       
)
demo.launch(inline=False)