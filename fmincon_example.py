import opt


def f(x):
    return x**2


def main():

    opt.start()

    x0 = (100, )
    lb = (-1000, )
    ub = (1000, )
    param, ll, exit_flat = opt.fmincon("fmincon_example.f",
                                       x0=x0, lb=lb, ub=ub)
    print(param)
    opt.stop()


if __name__ == "__main__":
    main()
