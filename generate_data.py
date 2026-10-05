import pandas as pd
import numpy as np
import time

# importing from the physics engine
from physics_engine import calc_performance, get_aircraft_data

print("-" * 60)
print("Generating flight data for ML training")
print("-" * 60)

# how many rows
n = 10000

# the 3 planes we have
ac_ids = [1, 2, 3]

print(f"\ngenerating {n} rows...")
print("this takes about 10 seconds\n")

rows = []
t0 = time.time()

for i in range(n):
    # pick a random plane
    ac_id = np.random.choice(ac_ids)
    
    # get plane info from db
    ac = get_aircraft_data(ac_id)
    
    # random values
    alt = np.random.uniform(1000, 12000)
    wt = np.random.uniform(ac['empty_weight_lbs'] + 50, ac['max_gross_weight_lbs'])
    wind = np.random.uniform(-20, 50)
    alpha = np.random.uniform(0, 15)
    
    try:
        res = calc_performance(
            aircraft_id=ac_id,
            altitude_ft=alt,
            alpha_deg=alpha,
            weight_lbs=wt,
            wind_kts=wind
        )
        
        # adding aircraft features for the prof's suggestion
        # using these instead of just aircraft_id so model can generalize
        rows.append({
            'aircraft_id': ac_id,
            'altitude_ft': alt,
            'weight_lbs': wt,
            'wind_kts': wind,
            'alpha_deg': alpha,
            'wing_area_sqft': ac['wing_area_sqft'],
            'wing_span_ft': ac['wing_span_ft'],
            'aspect_ratio': ac['aspect_ratio'],
            'max_gross_weight_lbs': ac['max_gross_weight_lbs'],
            'cl': res['cl'],
            'cd': res['cd'],
            'glide_ratio': res['glide_ratio_air'],
            'v_airspeed_kts': res['v_airspeed_kts'],
            'dist_ground_nm': res['dist_ground_nm'],
            'penalty_pct': res['penalty_pct']
        })
    except:
        # skip if something fails, shouldn't happen
        pass
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1} done")

# save it
df = pd.DataFrame(rows)
df.to_csv('data/synthetic_flight_data.csv', index=False)

elapsed = time.time() - t0

print(f"\ndone! {len(df)} rows saved")
print(f"took {elapsed:.1f} seconds")
print(f"columns: {df.columns.tolist()}")
print("\nfirst 5 rows:")
print(df.head())
print("=" * 60)