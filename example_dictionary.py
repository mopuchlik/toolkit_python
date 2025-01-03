# %% iterate through keys and values
file_counts = {"jpg": 10, "txt": 14, "csv": 2, "py": 23}
for ext, amount in file_counts.items():
    print("There are {} files with the .{} extension".format(amount, ext))


# %%
file_counts.keys()

# %%
file_counts.values()

# %% change single value
file_counts["jpg"]
file_counts["jpg"] = 100
file_counts

# %% add/change multiple values
file_counts.update({"jpg": 10, "xxx": 2, "avi": 4})
file_counts

# %% remove key and value
del file_counts["avi"]


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
# example switch keys with values


def groups_per_user(group_dictionary):
    user_groups = {}
    # Go through group_dictionary
    for groups, users in group_dictionary.items():
        # Now go through the users in the group
        for user in users:
            # Now add the group to the the list of
            # groups for this user, creating the entry
            # in the dictionary if necessary
            if user in user_groups:
                user_groups[user].append(groups)
            else:
                user_groups.update({user: [groups]})

    return user_groups


# %%
groups_per_user(
    {
        "local": ["admin", "userA"],
        "public": ["admin", "userB"],
        "administrator": ["admin"],
    }
)

# %%
wardrobe = {"shirt": ["red", "blue", "white"], "jeans": ["blue", "black"]}
new_items = {"jeans": ["white"], "scarf": ["yellow"], "socks": ["black", "brown"]}
wardrobe.update(new_items)
# %%
