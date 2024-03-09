from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import os
import requests

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2txt(img_path):
    # Load the model
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text = image_to_text(
        img_path)[0]["generated_text"]
    
    print(text)
    return text


#llm
def generate_scene(scenario):
    template = """
    You are describing the image to a blind person;
    The description is not more than 100 words;
    
    IMAGE: {scenario}
    """
    
    prompt = PromptTemplate(template =template, input_variables={"scenario"})
    
    descrption_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature =1), prompt= prompt, verbose=True)
    
    descrption = descrption_llm.predict(scenario=scenario)
    
    print(descrption)
    return descrption

description = img2txt("Screenshot_7.png")
elaborate = generate_scene(description)
