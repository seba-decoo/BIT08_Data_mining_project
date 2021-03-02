#test replace function on dataset

from os import sep
import pandas

ds = pandas.read_csv("heart.csv")

print(ds)

ds["sex"].replace({0:"F", 1:"M"}, inplace=True)
ds["cp"].replace({0:"typ_ang", 1:"atyp_ang", 2:"non-ang_pain", 3:"asymp"}, inplace= True)
ds["fbs"].replace({1:True, 0:False}, inplace=True)
ds["restecg"].replace({0:"normal", 1:"ST-T", 2:"LV_hyper"}, inplace= True)
ds["exang"].replace({0:"no", 1:"yes"}, inplace=True)
ds["slope"].replace({0:"UP", 1:"FL", 2:"DW" }, inplace= True)
ds["thal"].replace({1:"normal", 2:"fix_defect", 3:"revers_defect" }, inplace= True)
ds["target"].replace({0:"less_change", 1:"more_change"}, inplace=True)


print(ds)

ds.to_csv("heart_preprocessed.csv", sep=",", index=False)



