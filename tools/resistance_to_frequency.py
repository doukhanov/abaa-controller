"""Convert junction resistance targeting precision into qubit frequency precision.

Chain of relations
------------------
Ambegaokar-Baratoff      I_c = pi * Delta / (2 e R_n)
Josephson energy         E_J = hbar I_c / (2 e)
Transmon (E_J >> E_C)    f_01 ~ (sqrt(8 E_J E_C) - E_C) / h

Because E_J is proportional to 1/R_n, f_01 scales as R_n^(-1/2), so a
fractional error in resistance costs about half as much in frequency:

    df/f  ~  (1/2) * (dR/R) * (1 + E_C / h f_01)

Reported power-law exponents in the literature sit near -0.5 (e.g. -0.55 in
the LASIQ laser-annealing work), so treat this as a good estimate rather than
an exact conversion. For a SQUID, R_n is the parallel resistance of the two
junctions.

Usage
-----
    python tools/resistance_to_frequency.py --rn 6000 --tol 50
    python tools/resistance_to_frequency.py --rn 6000 --tol 50 --ec 250 --scan
"""

from __future__ import annotations

import argparse
import math

H = 6.62607015e-34          # J s
HBAR = H / (2 * math.pi)
E = 1.602176634e-19         # C
KB = 1.380649e-23           # J/K
BCS = 1.764                 # Delta = 1.764 kB Tc


def gap_joules(tc: float) -> float:
    """BCS gap from critical temperature."""
    return BCS * KB * tc


def critical_current(rn: float, delta: float) -> float:
    """Ambegaokar-Baratoff, zero temperature."""
    return math.pi * delta / (2 * E * rn)


def ej_over_h(rn: float, delta: float) -> float:
    """Josephson energy in Hz."""
    return HBAR * critical_current(rn, delta) / (2 * E) / H


def f01(rn: float, ec_hz: float, delta: float) -> float:
    """Transmon 0->1 transition frequency in Hz."""
    return math.sqrt(8 * ej_over_h(rn, delta) * ec_hz) - ec_hz


def report(rn: float, tol: float, ec_hz: float, delta: float) -> None:
    f0 = f01(rn, ec_hz, delta)
    df = abs(f01(rn + tol, ec_hz, delta) - f0)
    print(f"  R_n         {rn/1e3:8.2f} kOhm  +/- {tol:.0f} Ohm  ({tol/rn*100:.2f} %)")
    print(f"  I_c         {critical_current(rn, delta)*1e9:8.1f} nA")
    print(f"  E_J/h       {ej_over_h(rn, delta)/1e9:8.2f} GHz     E_J/E_C = {ej_over_h(rn, delta)/ec_hz:.0f}")
    print(f"  f_01        {f0/1e9:8.3f} GHz")
    print(f"  -> spread   {df/1e6:8.1f} MHz    ({df/f0*100:.3f} % of f_01)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rn", type=float, default=6000.0, help="normal-state resistance, Ohm")
    ap.add_argument("--tol", type=float, default=50.0, help="targeting tolerance, Ohm")
    ap.add_argument("--ec", type=float, default=200.0, help="charging energy E_C/h, MHz")
    ap.add_argument("--tc", type=float, default=1.2, help="film critical temperature, K")
    ap.add_argument("--scan", action="store_true", help="sweep R_n across the working range")
    args = ap.parse_args()

    delta = gap_joules(args.tc)
    ec_hz = args.ec * 1e6

    print(f"\nAl gap  Delta = {delta/E*1e6:.1f} ueV   (Tc = {args.tc} K)")
    print(f"E_C/h = {args.ec:.0f} MHz\n")
    report(args.rn, args.tol, ec_hz, delta)

    if args.scan:
        print(f"\n  {'R_n (kOhm)':>11}{'f_01 (GHz)':>12}{'dR/R (%)':>10}{'df (MHz)':>10}{'df/f (%)':>10}")
        print("  " + "-" * 53)
        lo, hi = args.rn / 2, args.rn * 1.5
        for i in range(9):
            rn = lo + (hi - lo) * i / 8
            f0 = f01(rn, ec_hz, delta)
            df = abs(f01(rn + args.tol, ec_hz, delta) - f0)
            print(f"  {rn/1e3:>11.2f}{f0/1e9:>12.3f}{args.tol/rn*100:>10.2f}"
                  f"{df/1e6:>10.1f}{df/f0*100:>10.3f}")
    print()


if __name__ == "__main__":
    main()
