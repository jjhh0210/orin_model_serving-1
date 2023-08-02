from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#from torch.quantization import quantize_dynamic
import threading

app = FastAPI()

class PredictRequest(BaseModel):
    corrected_content: str

model_name = 'jian1114/jian_KoBART_fine_tuned'
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

    return {'subheadings' : subheadings, 'paragraphs' : paragraph_text}

def process_paragraph(i, paragraph, subheadings):
    input_ids = tokenizer.encode(paragraph, return_tensors='pt', max_length=1024)
    
    with lock:
        output = model.generate(input_ids, max_length=32, num_beams=10, early_stopping=True)
        
    subheading = tokenizer.decode(output[0], skip_special_tokens=True)

    subheading_final = ""  # 실제 반환할 소제목
    check_list = ["em class", "violet_text", "green_text", "red_text","blue_text"]
    if  subheading=="O" or "OO" in subheading:
        subheading_final = "😢소제목 생성 실패: 더 자세한 내용이 필요합니다." 
    elif any(x in subheading for x in check_list):
        subheading_final = "😢소제목 생성 실패: 문법 교정 후 다시 시도해 보세요."
    else:
        subheading_final = subheading
    
    subheadings[i] = subheading_final
    

