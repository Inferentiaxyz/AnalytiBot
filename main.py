import pandas as pd
import matplotlib.pyplot as plt
import openai
import chainlit as cl
import re

system_prompt = """Act as a data scientist. Given a pandas dataframe called "df" you will help the user to make an exploratory analysis. 
You have to provide the python code for the data visualizations by using matplotlib and pandas with also the explaination of the analysis. 
The code has to save the figure of the visualization in an image called img.png without doing the plot.show(). The pandas dataframe is already loaded in the variable "df"
Columns of "df" dataframe: [{}]"""


openai.api_key_path = "openaikey.txt"
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": 1,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

df = pd.read_csv("./adult_csv.csv")

def get_dt_columns_info(df):
    # Get the column names and their value types
    column_types = df.dtypes
    # Convert the column_types Series to a list
    column_types_list = column_types.reset_index().values.tolist()
    infos = ""
    # Print the column names and their value types
    for column_name, column_type in column_types_list:
        infos+="{}:{}".format(column_name, column_type)
    return infos

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_prompt.format(get_dt_columns_info(df))}],
    )

def extract_code(gpt_response):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, gpt_response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message}) 
    # Generation of the image
    
    # Response of the LLM model
    response = openai.ChatCompletion.create(
        model=model_name, messages=message_history, stream=False, **settings
    )
    #GPT response
    gpt_response = response['choices'][0]['message']['content']
    print(gpt_response)

    # GPT CODE
    just_code = extract_code(gpt_response)
    exec(just_code)

    # Read the image
    elements = [
        cl.Image(name="image1", display="inline", path="./img.png")
    ]

    # Provide the explaination
    explaination = gpt_response.split("```")[2]
    await cl.Message(content=explaination, elements=elements).send()
