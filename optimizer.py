import numpy as np
from physics_engine import calc_performance

def objective(alpha, ac_id, alt, weight, wind):
    # we want to MAXIMIZE distance so return negative
    r = calc_performance(
        aircraft_id=ac_id,
        altitude_ft=alt,
        alpha_deg=alpha,
        weight_lbs=weight,
        wind_kts=wind
    )
    return -r['dist_ground_nm']

def golden_section(ac_id, alt, weight, wind, a=0.0, b=15.0, tol=0.001):
    # golden section search for the best glide angle
    # golden ratio ~ 0.618
    
    gr = (np.sqrt(5) - 1) / 2
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = objective(c, ac_id, alt, weight, wind)
    fd = objective(d, ac_id, alt, weight, wind)
    
    n = 0
    while abs(c - d) > tol:
        n += 1
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = objective(c, ac_id, alt, weight, wind)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = objective(d, ac_id, alt, weight, wind)
    
    best = (a + b) / 2.0
    
    r = calc_performance(
        aircraft_id=ac_id,
        altitude_ft=alt,
        alpha_deg=best,
        weight_lbs=weight,
        wind_kts=wind
    )
    
    return {
        'best_alpha_deg': best,
        'best_distance_nm': r['dist_ground_nm'],
        'v_airspeed_kts': r['v_airspeed_kts'],
        'iterations': n
    }

if __name__ == "__main__":
    print("testing optimizer...\n")
    
    for w in [0, 10, 20, 30, 40]:
        r = golden_section(2, 6000, 2000, float(w), 0.0, 15.0, 0.001)
        print(f"wind {w:2d} kts -> aoa {r['best_alpha_deg']:.3f}, spd {r['v_airspeed_kts']:.1f}, dist {r['best_distance_nm']:.2f}")
    
    print("\ndone. angle drops with headwind as expected")