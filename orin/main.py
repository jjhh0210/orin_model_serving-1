from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from torch.quantization import quantize_dynamic
import threading

app = FastAPI()

class PredictRequest(BaseModel):
    corrected_content: str

model_name = './kobart_model'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
#model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
lock = threading.Lock()  # This is for synchronizing access to the model

@app.post("/subhead/")
async def subheading_KoBART(request: PredictRequest):
    #torch.set_num_threads(1)

    paragraph_text = request.corrected_content.split("//")
    paragraph_text = [s.strip('\r\n') for s in paragraph_text]
    subheadings = [None] * len(paragraph_text)

    threads = []
    for i, paragraph in enumerate(paragraph_text):
        t = threading.Thread(target=process_paragraph, args=(i, paragraph, subheadings))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return {'subheadings' : dict(zip(subheadings, paragraph_text))}

def process_paragraph(i, paragraph, subheadings):
    input_ids = tokenizer.encode(paragraph, return_tensors='pt', max_length=1024)
    
    with lock:
        output = model.generate(input_ids, max_length=32, num_beams=10, early_stopping=True)
        
    subheading = tokenizer.decode(output[0], skip_special_tokens=True)
    subheadings[i] = subheading

