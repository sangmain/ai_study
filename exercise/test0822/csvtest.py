import pandas as pd

# time_pd = pd.DataFrame(pd.read_csv("./data/test0822.csv"))
# time_pd['date'] = None
# time_pd.to_csv("filename.csv", index=False)

time_pd = pd.DataFrame()
time_pd['Data'] = 'aaa'
time_pd.to_csv("filename.csv", index=False)