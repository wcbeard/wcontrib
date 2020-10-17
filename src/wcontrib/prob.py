import scipy.stats as sts


def compare_tf(df, ps=[.04, .96]):
    """
    TODO: compute MI row by row, one vs all
    """
    ps = list(ps)
    kwa = {f"p{int(p * 100):02}": lambda x: sts.beta(x.a, x.b).ppf(p) for p in ps}
    df = df.assign(
        a=lambda x: x[True] + 1,
        b=lambda x: x[False] + 1,
        tot=lambda x: x[True] + x[False],
    ).assign(**kwa)
    return df
