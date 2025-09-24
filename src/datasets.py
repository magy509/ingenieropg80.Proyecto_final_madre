import pandas as pd
import requests
import time
import gzip
import os

##primer dataset
def peli_info_df():
    informacion = ["links", "movies", "ratings", "tags"]
    df_list = []

    for i in informacion:
        df = pd.read_csv(f"../data/interim/movies-data/ml-32m/{i}.csv")
        df_list.append(df)
    
    
    return df_list

##segundo dataset
def feel_df(max_reviews=10000):
    file_path = "../data/interim/movies.txt.gz"
    data = []
    current_review = {}

    with gzip.open(file_path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("product/productId:"):
                if current_review:
                    data.append(current_review)
                    if len(data) >= max_reviews:
                        break  # se para al llegar al limite
                    current_review = {}
                current_review["productId"] = line[len("product/productId: "):]
            elif line.startswith("review/userId:"):
                current_review["userId"] = line[len("review/userId: "):]
            elif line.startswith("review/profileName:"):
                current_review["profileName"] = line[len("review/profileName: "):]
            elif line.startswith("review/helpfulness:"):
                current_review["helpfulness"] = line[len("review/helpfulness: "):]
            elif line.startswith("review/score:"):
                try:
                    current_review["score"] = float(line[len("review/score: "):])
                except:
                    current_review["score"] = None
            elif line.startswith("review/time:"):
                try:
                    current_review["time"] = int(line[len("review/time: "):])
                except:
                    current_review["time"] = None
            elif line.startswith("review/summary:"):
                current_review["summary"] = line[len("review/summary: "):] if len(line) > len("review/summary: ") else ""
            elif line.startswith("review/text:"):
                current_review["text"] = line[len("review/text: "):] if len(line) > len("review/text: ") else ""

        if current_review and len(data) < max_reviews:
            data.append(current_review)

    df = pd.DataFrame(data)
    print(f"Se cargaron {len(df)} reseñas")
    return df

##tercer dataset
def get_dataset(link):

    df = pd.read_csv(link)
    return df

## train y test de reseñas

def cargar_reseñas(directorio):
    textos = []
    etiquetas = []

    for label in ['pos', 'neg']:
        path = os.path.join(directorio, label)
        for archivo in os.listdir(path):
            with open(os.path.join(path, archivo), encoding='utf-8') as f:
                textos.append(f.read())
                etiquetas.append(1 if label == 'pos' else 0)
    return textos, etiquetas

