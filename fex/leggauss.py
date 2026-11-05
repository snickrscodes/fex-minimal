import math


def _ts_mult(u: list[float], h: float, n: int) -> float:
    # evaluates a polynomial: u[1] + u[2]*h + ... + u[n]*h^(n-1)
    ts = 0.0
    hk = 1.0
    for k in range(1, n + 1):
        ts += u[k] * hk
        hk *= h
    return ts


def _rk2_leg(t1: float, t2: float, x: float, n: int) -> float:
    # advances the value of x(t) using runge kutta (second order) method
    m = 10
    h = (t2 - t1) / m
    snn1 = math.sqrt(n * (n + 1))
    t = t1

    for _ in range(m):
        f = (1.0 - x) * (1.0 + x)
        k1 = -h * f / (snn1 * math.sqrt(f) - 0.5 * x * math.sin(2.0 * t))
        x = x + k1
        t = t + h
        f = (1.0 - x) * (1.0 + x)
        k2 = -h * f / (snn1 * math.sqrt(f) - 0.5 * x * math.sin(2.0 * t))
        x = x + 0.5 * (k2 - k1)

    return x


def _legendre_compute_glr0(n: int):
    # starting value for fast algorithm
    pm2, pm1 = 0.0, 1.0
    ppm2, ppm1 = 0.0, 0.0
    p, pp = 0.0, 0.0

    for k in range(n):
        dk = float(k)
        p = -dk * pm2 / (dk + 1.0)
        pp = ((2.0 * dk + 1.0) * pm1 - dk * ppm2) / (dk + 1.0)
        pm2 = pm1
        pm1 = p
        ppm2 = ppm1
        ppm1 = pp

    return p, pp


def _legendre_compute_glr2(pn0: float, n: int):
    # finds first real root (only when n is even)
    m = 30
    x1 = _rk2_leg(0.0, -math.pi / 2.0, 0.0, n)
    u = [0.0] * (m + 2)
    up = [0.0] * (m + 1)
    dn = float(n)
    u[1] = pn0

    for k in range(0, m - 1, 2):
        dk = float(k)
        u[k + 3] = (
            (dk * (dk + 1.0) - dn * (dn + 1.0))
            * u[k + 1]
            / ((dk + 1.0) * (dk + 2.0))
        )
        up[k + 2] = (dk + 2.0) * u[k + 3]

    # newton iterations
    for _ in range(5):
        x1 = x1 - _ts_mult(u, x1, m) / _ts_mult(up, x1, m - 1)
    d1 = _ts_mult(up, x1, m - 1)
    return x1, d1


def _legendre_compute_glr1(n: int, x: list[float], ders: list[float]):
    # gets complete set of legendre points and derivatives
    m = 30
    if n % 2 == 1:
        n2 = (n - 1) // 2
        s = 1
    else:
        n2 = n // 2
        s = 0

    u = [0.0] * (m + 2)
    up = [0.0] * (m + 1)
    dn = float(n)

    for j in range(n2, n - 1):
        xp = x[j]
        h = _rk2_leg(math.pi / 2.0, -math.pi / 2.0, xp, n) - xp
        u[0], u[1] = 0.0, 0.0
        u[2] = ders[j]
        up[0] = 0.0
        up[1] = u[2]

        for k in range(m - 1):
            dk = float(k)
            u[k + 3] = (
                2.0 * xp * (dk + 1.0) * u[k + 2]
                + (dk * (dk + 1.0) - dn * (dn + 1.0)) * u[k + 1] / (dk + 1.0)
            ) / ((1.0 - xp) * (1.0 + xp) * (dk + 2.0))
            up[k + 2] = (dk + 2.0) * u[k + 3]

        for _ in range(5):
            h = h - _ts_mult(u, h, m) / _ts_mult(up, h, m - 1)
        x[j + 1] = xp + h
        ders[j + 1] = _ts_mult(up, h, m - 1)

    # apply symmetry
    for k in range(n2 + s):
        x[k] = -x[n - k - 1]
        ders[k] = -ders[n - k - 1] if n % 2 == 0 else ders[n - k - 1]


def leggauss_glr(n: int):
    # returns (nodes, weights)
    x = [0.0] * n
    w = [0.0] * n
    ders = [0.0] * n
    # get value and derivative of nth legendre polynomial at 0
    p, pp = _legendre_compute_glr0(n)

    # set up the first root
    if n % 2 == 1:
        idx = (n - 1) // 2
        x[idx] = p
        ders[idx] = pp
    else:
        idx = n // 2
        x[idx], ders[idx] = _legendre_compute_glr2(p, n)

    # get the rest of the roots and derivatives
    _legendre_compute_glr1(n, x, ders)
    # compute weights
    for i in range(n):
        w[i] = 2.0 / ((1.0 - x[i]) * (1.0 + x[i]) * (ders[i] ** 2))

    # normalize weights to sum to 2.0
    w_sum = sum(w)
    w = [2.0 * wi / w_sum for wi in w]
    return x, w
