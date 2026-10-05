import sqlite3
import pandas as pd
import numpy as np

# db path - hardcoded because os.path.join was giving me grief
db = 'data/glide.db'

# caching so we don't hammer the db every single call
polar_cache = {}
ac_cache = {}

def get_polar_data(ac_id):
    # loads cl/cd data for one aircraft
    if ac_id in polar_cache:
        return polar_cache[ac_id]
    
    con = sqlite3.connect(db)
    q = f"SELECT alpha_deg, cl, cd FROM airfoil_polars WHERE aircraft_id = {ac_id} ORDER BY alpha_deg"
    d = pd.read_sql_query(q, con)
    con.close()
    
    d['glide_ratio'] = d['cl'] / d['cd']
    
    polar_cache[ac_id] = d
    return d

def get_aircraft_data(ac_id):
    if ac_id in ac_cache:
        return ac_cache[ac_id]
    
    con = sqlite3.connect(db)
    q = f"SELECT * FROM aircraft WHERE id = {ac_id}"
    d = pd.read_sql_query(q, con)
    con.close()
    
    if len(d) > 0:
        ac_cache[ac_id] = d.iloc[0]
        return ac_cache[ac_id]
    
    return None

def get_max_glide(ac_id):
    d = get_polar_data(ac_id)
    i = d['glide_ratio'].idxmax()
    return d.loc[i, 'glide_ratio'], d.loc[i, 'alpha_deg']

def get_cl_cd(ac_id, alpha):
    # interpolate cl and cd if the angle isn't exactly in the data
    d = get_polar_data(ac_id)
    
    # exact match, easy
    if alpha in d['alpha_deg'].values:
        r = d[d['alpha_deg'] == alpha].iloc[0]
        return r['cl'], r['cd']
    
    # sort so interpolation works
    ds = d.sort_values('alpha_deg').reset_index(drop=True)
    
    # clamp to edges if out of range
    if alpha < ds['alpha_deg'].min():
        r = ds.iloc[0]
        return r['cl'], r['cd']
    
    if alpha > ds['alpha_deg'].max():
        r = ds.iloc[-1]
        return r['cl'], r['cd']
    
    # find surrounding points
    lo = ds[ds['alpha_deg'] <= alpha].index[-1]
    hi = ds[ds['alpha_deg'] >= alpha].index[0]
    
    if lo == hi:
        r = ds.iloc[lo]
        return r['cl'], r['cd']
    
    L = ds.iloc[lo]
    U = ds.iloc[hi]
    
    # simple linear interp, good enough for this
    a1, cl1, cd1 = L['alpha_deg'], L['cl'], L['cd']
    a2, cl2, cd2 = U['alpha_deg'], U['cl'], U['cd']
    
    cl = cl1 + (cl2 - cl1) * (alpha - a1) / (a2 - a1)
    cd = cd1 + (cd2 - cd1) * (alpha - a1) / (a2 - a1)
    
    return cl, cd

def calc_performance(aircraft_id, altitude_ft, alpha_deg, weight_lbs, wind_kts):
    # main function. speed comes from lift equation
    # V = sqrt(2W / (rho * S * CL))
    # originally used POH speed but that was wrong
    
    RHO = 0.0023769   # slug/ft^3
    K2F = 1.68781     # knots to fps
    F2NM = 6076       # feet to nm
    
    ac = get_aircraft_data(aircraft_id)
    if ac is None:
        raise ValueError(f"aircraft {aircraft_id} not in db")
    
    S = ac['wing_area_sqft']
    
    cl, cd = get_cl_cd(aircraft_id, alpha_deg)
    gr = cl / cd
    
    max_gr, best_a = get_max_glide(aircraft_id)
    
    # this is the important bit - speed depends on weight and cl
    v_fps = np.sqrt((2 * weight_lbs) / (RHO * S * cl))
    v_kts = v_fps / K2F
    
    # sink rate
    sink = v_fps * (cd / cl)
    
    # time and distance
    t_sec = altitude_ft / sink
    t_min = t_sec / 60.0
    
    d_air = (v_fps * t_sec) / F2NM
    
    # wind adjustment
    w_fps = wind_kts * K2F
    g_fps = v_fps - w_fps
    
    if g_fps <= 0:
        d_gnd = 0.0
        msg = "wind too strong"
    else:
        d_gnd = (g_fps * t_sec) / F2NM
        msg = f"dist: {d_gnd:.2f} nm"
    
    # penalty vs best
    pen = ((max_gr - gr) / max_gr) * 100.0
    
    return {
        'cl': cl,
        'cd': cd,
        'glide_ratio_air': gr,
        'max_glide_ratio': max_gr,
        'best_alpha_deg': best_a,
        'v_airspeed_kts': v_kts,
        'v_sink_fps': sink,
        'time_minutes': t_min,
        'dist_air_nm': d_air,
        'dist_ground_nm': d_gnd,
        'penalty_pct': pen,
        'message': msg
    }

# quick test
if __name__ == "__main__":
    r = calc_performance(2, 6000, 5.0, 2000, 10)
    print(f"cl={r['cl']:.4f}, cd={r['cd']:.5f}")
    print(f"gr={r['glide_ratio_air']:.2f}, spd={r['v_airspeed_kts']:.1f} kts")
    print(f"dist={r['dist_ground_nm']:.2f} nm")
    print("ok")