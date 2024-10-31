import json
import pandas as pd
import os

dataset = pd.read_csv('./dataset.csv')
filestem_col = dataset['filestem']
fracture_col = dataset['fracture_visible']
fracture_col = fracture_col.fillna(0)

def main():
    json_data = []
    for i in range(len(filestem_col)):
        json_data.append({
            "source" : f"{filestem_col[i]}",
            "segment" : f"{filestem_col[i]}",
            "condition" : fracture_col[i]
        })

    with open("data.json", "w") as f:
        json.dump(json_data, f, indent=4)
        f.write('\n')

if __name__ == "__main__":
    main()