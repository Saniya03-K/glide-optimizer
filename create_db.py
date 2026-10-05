import sqlite3
import pandas as pd

# hardcoding paths because i'm tired of os.path.join
db_path = 'data/glide.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 60)
print("Creating tables...")
print("=" * 60)

# drop first, i've rerun this so many times
cursor.execute('DROP TABLE IF EXISTS aircraft;')

cursor.execute('''
CREATE TABLE aircraft (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    best_glide_kias INTEGER,
    empty_weight_lbs INTEGER,
    max_gross_weight_lbs INTEGER,
    wing_area_sqft REAL,
    wing_span_ft REAL,
    aspect_ratio REAL,
    ref_cl_over_cd REAL,
    ref_reynolds INTEGER
)
''')

# wing spans from the POH, aspect ratio = span^2 / area
planes = [
    ('Cessna 152', 60, 1080, 1670, 160.0, 33.33, 6.94, 9.0, 200000),
    ('Cessna 172S', 68, 1669, 2550, 174.0, 36.08, 7.48, 9.1, 500000),
    ('Piper Cherokee PA-28-140', 73, 1250, 2150, 160.0, 30.00, 5.63, 8.8, 1000000)
]

# just looping, executemany felt overkill
for p in planes:
    cursor.execute('''
        INSERT INTO aircraft 
        (name, best_glide_kias, empty_weight_lbs, max_gross_weight_lbs, wing_area_sqft, wing_span_ft, aspect_ratio, ref_cl_over_cd, ref_reynolds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', p)

print("planes inserted")

# airfoil table
cursor.execute('DROP TABLE IF EXISTS airfoil_polars;')
cursor.execute('''
CREATE TABLE airfoil_polars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aircraft_id INTEGER,
    alpha_deg REAL,
    cl REAL,
    cd REAL,
    reynolds_number INTEGER
)
''')

# ok this part was annoying. airfoiltools csv has 10 rows of junk at top
# and the column names have spaces sometimes

re_map = {200000: 1, 500000: 2, 1000000: 3}

files = [
    ('naca2412_re200k.csv', 200000),
    ('naca2412_re500k.csv', 500000),
    ('naca2412_re1m.csv', 1000000)
]

for fname, re in files:
    path = 'data/' + fname
    df = pd.read_csv(path, skiprows=10)
    
    # cleaning up column names
    df.columns = df.columns.str.strip()
    
    # i only need these 3 columns, the rest i don't care about
    # had to rename because sometimes its 'Alpha' sometimes 'alpha '
    new_cols = {}
    for c in df.columns:
        if c.lower() == 'alpha':
            new_cols[c] = 'a'
        elif c.lower() == 'cl':
            new_cols[c] = 'cl'
        elif c.lower() == 'cd':
            new_cols[c] = 'cd'
    df = df.rename(columns=new_cols)
    
    # only want positive lift
    df = df[df['cl'] > 0]
    
    ac_id = re_map[re]
    
    # this loop is slow but whatever, its only 80 rows
    for i in range(len(df)):
        row = df.iloc[i]
        cursor.execute('''
            INSERT INTO airfoil_polars (aircraft_id, alpha_deg, cl, cd, reynolds_number)
            VALUES (?, ?, ?, ?, ?)
        ''', (ac_id, float(row['a']), float(row['cl']), float(row['cd']), re))
    
    print(f"done with {fname}")

conn.commit()
conn.close()

print("=" * 60)
print("done. db is at data/glide.db")
print("=" * 60)