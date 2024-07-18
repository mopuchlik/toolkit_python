# %%
import pandas as pd


def fun_example(data):

    x = data.loc[data["b"] >= 5]
    data["d"] = data["a"] + data["b"]

    return x


def fun_example_copy(data):

    data1 = data.copy()

    x = data1.loc[data1["b"] >= 5]
    data1["d"] = data1["a"] + data1["b"]

    return x


# %%

data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
data = pd.DataFrame(data)

x = fun_example(data)

# changes in data are preserved as there is no copy
# slice in x makes a copy on its own
print(x)
print(data)


# %%
data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

data = pd.DataFrame(data)

x = fun_example_copy(data)

# changes in data are not preserved
# slice in x makes a copy on its own
print(x)
print(data)
# %%
