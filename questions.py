#%%
import time


#%%
def decorator(func):
    """
    Decorator dedicated to measure a time run of decorated method.
    It is printing in IDE console how many seconds the particular method was running.

    Example:

    @decorator
    def multiply(number):
        time.sleep(4)
        return number * 10
    multiply(12)

    On console should appear for : "multiply: 4.01s"
    (time 4.01 is an example)
    """
    ### TO BYLO DO NAPISANIA A JA NIE NAPISALEM
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper

#%%

@decorator
def multiply(number):
    time.sleep(4)
    return number * 10

multiply(12)

#%% #############################################################

lst = []
def sum_list(num, lst = []):

    lst.append(num)
    x = sum(lst)

    return x
#%% GUESS WHAT IS THE ANSWER
sum_list(1)

#%% GUESS WHAT IS THE ANSWER
sum_list(2)

# %%  GUESS WHAT IS THE ANSWER
sum_list(3, [])

# %% FIX

lst = []
def sum_list2(num, lst = None):

    # init
    lst = []
    lst.append(num)
    x = sum(lst)

    return x

#%%
sum_list2(1)

#%%
sum_list2(2)

# %%
sum_list2(3)

