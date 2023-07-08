import pandas as pd

system_prompt = """Act as a data scientist. Given a pandas dataframe called "df" you will help the user to make an exploratory analysis. 
You have to provide the code for the data visualizations with also the explainations.

Columns of "df" dataframe: [{}]"""

df = pd.read_csv("./adult_csv.csv")

# Get the column names and their value types
column_types = df.dtypes

# Convert the column_types Series to a list
column_types_list = column_types.reset_index().values.tolist()

# Print the column names and their value types
for column_name, column_type in column_types_list:
    print(f"{column_name}: {column_type}")


import matplotlib.pyplot as plt

#Code
plt.hist(df['age'], bins=10, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
