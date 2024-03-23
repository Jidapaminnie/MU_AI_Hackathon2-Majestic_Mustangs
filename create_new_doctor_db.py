import pandas as pd
import uuid

df = pd.read_csv("data/siriraj_doctor_details.csv")
df['doctorID'] = [i for i in range(len(df.index))]
df.to_csv("data/test.csv", index=False)