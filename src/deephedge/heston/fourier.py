"""Three Fourier-based pricers for European options under any model with a known
characteristic function (here Heston).

    heston_price_integral  Gil-Pelaez / Heston's P1-P2 form, adaptive quadrature.
                           Slow but very accurate: the reference value.
    carr_madan_fft         Carr & Madan (1999). One FFT prices a whole grid of strikes.
    cos_price              Fang & Oosterlee (2008). Cosine series; exponential convergence,
                           the fastest here and what calibration uses.

Calls and puts are linked by put-call parity: C - P = S e^{-qT} - K e^{-rT}.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .model import HestonParams, cumulants, heston_cf


def _parity(call_price, S, K, T, r, q, call):
    return call_price if call else call_price - S * np.exp(-q * T) + K * np.exp(-r * T)


# --------------------------------------------------------------------- 1. Gil-Pelaez

def heston_price_integral(S, K, T, r, p: HestonParams, q=0.0, call=True, upper=np.inf):
    """C = S e^{-qT} P1 - K e^{-rT} P2, with

        P2 = Q(S_T > K)            = 1/2 + 1/pi int_0^inf Re[e^{-iu k} phi(u) / (iu)] du
        P1 = Q^S(S_T > K) (stock measure) = same with phi(u - i) / phi(-i)

    where k = log(K/S) and phi is the characteristic function of log(S_T/S).
    """
    k = np.log(K / S)
    fwd = heston_cf(-1j, T, p, r, q).real  # = e^{(r-q)T}

    def integrand(u, j):
        num = heston_cf(u - 1j, T, p, r, q) / fwd if j == 1 else heston_cf(u, T, p, r, q)
        return (np.exp(-1j * u * k) * num / (1j * u)).real

    P1 = 0.5 + quad(integrand, 1e-10, upper, args=(1,), limit=500, epsabs=1e-12)[0] / np.pi
    P2 = 0.5 + quad(integrand, 1e-10, upper, args=(2,), limit=500, epsabs=1e-12)[0] / np.pi
    c = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return _parity(c, S, K, T, r, q, call)


# --------------------------------------------------------------------- 2. Carr-Madan

def carr_madan_fft(S, K, T, r, p: HestonParams, q=0.0, call=True,
                   alpha=1.5, N=4096, eta=0.25):
    """Carr-Madan FFT.

    The call price as a function of log-strike k is not integrable (it tends to S as
    k -> -inf), so multiply by e^{alpha k} first. The damped price has the Fourier
    transform

        psi(u) = e^{-rT} phi_T(u - (alpha + 1) i) / (alpha^2 + alpha - u^2 + i (2 alpha + 1) u)

    with phi_T the cf of log S_T. Discretising the inverse transform on u_j = j eta and
    k_m = -b + m lambda with lambda eta = 2 pi / N turns it into one FFT. Simpson weights
    make the quadrature more accurate. Strikes in between are interpolated.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    u = np.arange(N) * eta
    k0 = np.log(S) - b  # centre the log-strike grid on log(S)
    k_grid = k0 + lam * np.arange(N)

    phi_T = heston_cf(u - (alpha + 1) * 1j, T, p, r, q) * np.exp(1j * (u - (alpha + 1) * 1j) * np.log(S))
    psi = np.exp(-r * T) * phi_T / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
    simpson = (3 + (-1) ** (np.arange(N) + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    x = np.exp(-1j * k0 * u) * psi * eta * simpson
    calls = (np.exp(-alpha * k_grid) / np.pi) * np.fft.fft(x).real

    c = np.interp(np.log(K), k_grid, calls)
    out = _parity(c, S, K, T, r, q, call)
    return out if out.size > 1 else float(out[0])


# --------------------------------------------------------------------- 3. COS

def _chi_psi(k, a, b, c, d):
    """Cosine-series coefficients of e^y and 1 on [c, d] (Fang & Oosterlee eq. 22-23)."""
    w = k * np.pi / (b - a)
    chi = (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
           + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w**2)
    psi = np.where(k == 0, d - c,
                   (np.sin(w * (d - a)) - np.sin(w * (c - a))) / np.where(k == 0, 1, w))
    return chi, psi


def cos_price(S, K, T, r, p: HestonParams, q=0.0, call=True, N=256, L=10.0):
    """Fang-Oosterlee COS method, vectorised over strikes.

    Idea: on a finite interval [a, b] that holds almost all the probability mass of
    y = log(S_T/K), expand the density in a cosine series. The series coefficients are
    (almost) the characteristic function sampled at u_k = k pi / (b - a), so pricing is

        V = K e^{-rT} sum'_{k<N} Re[ phi(u_k) e^{i u_k (x - a)} ] V_k,   x = log(S/K)

    where V_k are the cosine coefficients of the payoff (closed form). Error decays
    exponentially in N for smooth densities: N = 256 gives ~1e-8 here.

    The interval is centred on the expected value of y, x + c1, separately for each
    strike, with half-width L sqrt(c2 + sqrt(c4)).

    Puts are priced directly and calls recovered by parity, because the put payoff is
    bounded, so the result is far less sensitive to the choice of [a, b].
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    c1, c2, c4 = cumulants(T, p, r, q)
    width = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))  # Fang & Oosterlee eq. 49
    x = np.log(S / K)[:, None]                         # one row per strike
    a = x + c1 - width
    b = x + c1 + width

    k = np.arange(N)[None, :]
    u = k * np.pi / (b - a)
    # put payoff (1 - e^y)^+ is non-zero on y < 0, so integrate over [a, min(b, 0)]
    # (at short maturities a deep in-the-money put has the whole interval below 0, and a
    # deep out-of-the-money one has it all above 0, so the put is worth ~0)
    d = np.minimum(b, 0.0)
    chi, psi = _chi_psi(k, a, b, a, d)
    Vk = np.where(d > a, 2.0 / (b - a) * (-chi + psi), 0.0)  # cosine coeffs of (1 - e^y)^+

    phi = heston_cf(u, T, p, r, q)  # cf of log(S_T/S); the x shift goes in the exponent
    terms = (phi * np.exp(1j * u * (x - a))).real * Vk
    terms[:, 0] *= 0.5  # the ' in sum' : first term has weight 1/2
    puts = K * np.exp(-r * T) * terms.sum(axis=1)

    out = puts if not call else puts + S * np.exp(-q * T) - K * np.exp(-r * T)
    return out if out.size > 1 else float(out[0])
