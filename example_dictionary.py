# %% iterate through keys and values
file_counts = {"jpg": 10, "txt": 14, "csv": 2, "py": 23}
for ext, amount in file_counts.items():
    print("There are {} files with the .{} extension".format(amount, ext))


# %%
file_counts.keys()

# %%
file_counts.values()


# %% zlicza litery
def count_letters(text):
    result = {}
    for letter in text:
        if letter not in result:
            result[letter] = 0
        result[letter] += 1
    return result


# count_letters("aaaaa")
# count_letters("tenant")
count_letters("a long string with a lot of letters")

# %%
