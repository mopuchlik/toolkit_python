def f(*args, **kwargs):
    def g(*args, **kwargs):
        print("g:", args, kwargs)

    def h(*args, **kwargs):
        print("h:", args, kwargs)

    print("f:", args, kwargs)
    g(*args, **kwargs)  # Forward arguments to g()
    h(*args, **kwargs)  # Forward arguments to h()


f(10, 20, x=1, y=2)


# %%
def f(a, b, c, d, e):
    def g(a, c):
        print("g:", a, c)

    def h(b, d, e):
        print("h:", b, d, e)

    # Forward only relevant arguments to g()
    g(a=a, c=c)

    # Forward only relevant arguments to h()
    h(b=b, d=d, e=e)


f(a=1, b=2, c=3, d=4, e=5)


