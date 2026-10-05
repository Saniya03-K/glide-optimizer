# connects to flightgear and flies the plane
# spent way too long figuring out the telnet port thing

import time
import socket
from physics_engine import calc_performance, get_aircraft_data, get_max_glide
from optimizer import golden_section

HOST = "localhost"
PORT = 5401

def fg_get(prop):
    # read a property via telnet
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PORT))
        s.send(f"get {prop}\r\n".encode())
        r = s.recv(1024).decode()
        s.close()
        # response looks like: "/position/altitude-ft = '5000.0' (double)"
        if '=' in r:
            parts = r.split('=')
            if len(parts) > 1:
                v = parts[1].strip().split()[0].strip("'")
                return float(v)
    except Exception as e:
        print(f"read err {prop}: {e}")
    return None

def fg_set(prop, val):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PORT))
        s.send(f"set {prop} {val}\r\n".encode())
        r = s.recv(1024).decode()
        s.close()
        if prop in r and '=' in r:
            return True
        return False
    except Exception as e:
        print(f"write err {prop}: {e}")
        return False

def get_state():
    alt = fg_get("/position/altitude-ft")
    spd = fg_get("/velocities/airspeed-kt")
    pit = fg_get("/orientation/pitch-deg")
    wnd = fg_get("/environment/wind-speed-kt")
    
    # defaults if we got nothing back
    return {
        'alt': alt if alt is not None else 5000,
        'spd': spd if spd is not None else 70,
        'pitch': pit if pit is not None else 0,
        'wind': wnd if wnd is not None else 0
    }

def set_pitch(deg):
    # elevator-trim seems to work better than elevator
    return fg_set("/controls/flight/elevator-trim", deg / 10.0)

def show(state, opt):
    print("\n" + "=" * 60)
    print(f"Alt: {state['alt']:.0f} ft  |  Spd: {state['spd']:.1f} kts")
    print(f"Wind: {state['wind']:.1f} kts  |  Pitch: {state['pitch']:.1f} deg")
    print("-" * 60)
    print(f"Best AoA: {opt['best_alpha_deg']:.2f} deg")
    print(f"Distance: {opt['best_distance_nm']:.2f} NM")
    print("=" * 60)

def run(ac_id=2, weight=2000):
    print("=" * 60)
    print("FlightGear Autopilot (Ctrl+C to stop)")
    print("=" * 60)
    print(f"ac id: {ac_id}, weight: {weight} lbs")
    print(f"telnet port: {PORT}")
    print("\nwaiting for connection...")
    
    # test connection
    t = fg_get("/position/altitude-ft")
    if t is None:
        print("WARNING: cannot connect. check --telnet=5401")
    else:
        print(f"connected. alt = {t:.0f} ft")
    
    print("\nrunning loop...\n")
    
    try:
        while True:
            st = get_state()
            o = golden_section(
                ac_id=ac_id,
                alt=st['alt'],
                weight=weight,
                wind=st['wind'],
                a=0.0,
                b=15.0,
                tol=0.01
            )
            show(st, o)
            
            tgt = o['best_alpha_deg']
            ok = set_pitch(tgt)
            if ok:
                print(f"set pitch to {tgt:.2f}")
            else:
                print(f"FAILED to set pitch to {tgt:.2f}")
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopped.")

if __name__ == "__main__":
    run()