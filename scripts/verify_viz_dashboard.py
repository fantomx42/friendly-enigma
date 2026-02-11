import subprocess
import time
import requests
import sys
import os
import signal

def verify():
    print("--- Visualization & Diagnostics Phase 4 Verification (Dashboard) ---")
    
    # Start server
    print("1. Starting dashboard server...")
    # We use subprocess.Popen to start it non-blocking
    # We set env to ensure it uses a test storage if we wanted, 
    # but let's use default to check general start-up
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    # Using python -m wheeler.cli dashboard
    cmd = [sys.executable, "-m", "wheeler.cli", "dashboard"]
    
    # Create a process group so we can kill it easily
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid
    )
    
    # Wait for startup
    time.sleep(3)
    
    # Check if it crashed
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print("[FAIL] Server failed to start.")
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return

    try:
        print("2. Pinging http://localhost:5000/ ...")
        resp = requests.get("http://localhost:5000/")
        
        if resp.status_code == 200:
            print("   - [OK] Status 200")
            if "Wheeler Memory Dashboard" in resp.text:
                 print("   - [OK] Content matched title.")
            else:
                 print("   - [WARN] Content mismatch.")
        else:
            print(f"   - [FAIL] Status {resp.status_code}")
            
    except Exception as e:
        print(f"   - [FAIL] Connection error: {e}")
        
    finally:
        print("3. Stopping server...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()

    print("--- Done ---")

if __name__ == "__main__":
    verify()
