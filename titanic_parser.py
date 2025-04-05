import pandas as pd

def names(df):
    df.columns = df.columns.str.strip().str.lower()
    return [row['name'] for _, row in df.iterrows() if pd.notna(row['age']) and row['age'] < 33]

def birth_year_column(df):
    df['birth_year'] = df['age'].apply(lambda x: 1912 - x if pd.notnull(x) else None)
    return df

df = pd.read_csv("./assets/titanic.csv")

names_list = names(df)
print(names_list)

df_with_birth_year = birth_year_column(df)
print(df_with_birth_year[['name', 'birth_year']].head())
