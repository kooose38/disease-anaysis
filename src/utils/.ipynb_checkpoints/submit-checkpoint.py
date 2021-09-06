import pandas as pd 

def submit(pred: pd.Series, name: str):
    submit = pd.read_csv("./data/raw/test.csv")
    submit = submit[["id"]]
    submit["pred"] = pred 
    submit.to_csv(f"./data/submit/{name}0906_01.csv", index=False, header=False)