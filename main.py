import pandas as pd
import matplotlib.pyplot as plt
import openai
import chainlit as cl
import re
import chardet

system_prompt = """You are a great assistant at python data visualization creation.  You should create: the code for the data visualization in python using pandas and mathplotlib of a dataframe called "df".
Besides, Here are some requirements:
1: The pandas dataframe is already loaded in the variable "df".
2: Do not load the dataframe in the generated code!
2. The code has to save the figure of the visualization in an image called img.png do not do the plot.show().
3. Give the explainations along the code on how important is the visualization and what insights can we get
4. If the user asks for suggestions of analysis just provide the possible analysis without the code.
5. For any visualizations write only one block of code.
6. The available fields in the dataset "df" and their types are: {}"""


openai.api_key_path = "openaikey.txt"
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": 1,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

df = None

def get_dt_columns_info(df):
    # Get the column names and their value types
    column_types = df.dtypes
    # Convert the column_types Series to a list
    column_types_list = column_types.reset_index().values.tolist()
    infos = ""
    # Print the column names and their value types
    for column_name, column_type in column_types_list:
        infos+="{}({}),\n".format(column_name, column_type)
    return infos[:-1]

@cl.on_chat_start
async def start_chat():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload you csv/xlsx dataset file to begin!", accept=["csv","xlsx"], max_size_mb=100
        ).send()
    # Decode the file
    text_file = files[0]
    text = text_file.content
    f = open(text_file.path, "wb")
    f.write(text)
    f.close()
    global df
    if "csv" in text_file.path:
        df = pd.read_csv(text_file.path)
    else:
        df = pd.read_excel(text_file.path, index_col=0)    
    await cl.Message(
        content=f"`{text_file.name}` uploaded correctly!\n it contains {df.shape[0]} Rows and {df.shape[1]} Columns where each column type are:\n [{get_dt_columns_info(df)}]"
    ).send()

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_prompt.format(get_dt_columns_info(df))}],
    )

def extract_code(gpt_response):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[-1]
    else:
        return None
    
def filter_rows(text):
    # Split the input string into individual rows
    lines = text.split('\n')
    filtered_lines = [line for line in lines if "pd.read_csv" not in line and "pd.read_excel" not in line  and ".show()" not in line]
    filtered_text = '\n'.join(filtered_lines)
    
    return filtered_text

def interpret_code(gpt_response):
    if "```" in gpt_response:
        just_code = extract_code(gpt_response)
        if just_code.startswith("python"):
            just_code = just_code[len("python"):]
        just_code = filter_rows(just_code)
        print("CODE part:{}".format(just_code))
        # Interpret the code
        print("Codice da interpretare.")
        exec(just_code)
        return True
    else:
        return False

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
    print("GPT response:{}".format(gpt_response))

    # Extract code and interpret IT
    has_code = interpret_code(gpt_response)

    final_message = ""
    if has_code:
        # Read the image
        elements = [
            cl.Image(name="image1", display="inline", path="./img.png")
        ]

        # Provide the explaination
        final_message = gpt_response.split("```")[-1]
        await cl.Message(content=final_message, elements=elements).send()
    else:
        final_message = gpt_response
        await cl.Message(content=final_message).send()
    
