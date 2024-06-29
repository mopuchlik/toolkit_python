# %%
import itertools as it

# %%
iter_l = [1, 4, 7]
iterator = iter(iter_l)

print(next(iterator))
print(next(iterator))
print(next(iterator))

# %%
x = it.count(10)
print(next(x))
print(next(x))
print(next(x))
print(next(x))

# %%
x = it.cycle("ABC")
print(next(x))
print(next(x))
print(next(x))
print(next(x))


# %%
x = it.repeat("ABC", 1)
print(next(x))
print(next(x))


# %%

p = [1, 3, 5]
q = [1, 3, 5]

x = it.product(p, q)
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
# print(next(x))


# %%
p = [1, 3, 5]

x = it.permutations(p)
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))

# %%
p = [1, 3, 5]
q = [9, 10]

x = it.combinations(p, 2)
print(next(x))
print(next(x))
print(next(x))

# %%

ll = [1, 3, 6]
x = it.accumulate(ll)
print(next(x))
print(next(x))
print(next(x))


# %%

x = it.chain("ab", "cd", "f")
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))

# %%
ll = ["ab", "cd", "f"]
x = it.chain.from_iterable(ll)
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))


# %%
