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

# %% example of while loop

# Sample dictionary
my_dict = {"apple": 3, "banana": 5, "cherry": 2}
print(my_dict)

# While the dictionary is not empty
while my_dict:
    # Get and remove an item using popitem()
    key, value = my_dict.popitem()
    print(f"Processing {key}: {value}")

print(my_dict)

# %%
