import subprocess
import os
import sys

def run_command(args, env=None):
    cmd = [sys.executable, "-m", "wheeler.cli"] + args
    cmd_str = " ".join(args)
    print(f"\n> python -m wheeler.cli {cmd_str}")
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        env=env
    )
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
    return result

def verify():
    print("--- Phase 4 Manual Verification (CLI) ---")
    
    storage_dir = os.path.abspath("verification_cli_storage")
    env = os.environ.copy()
    env["WHEELER_STORAGE"] = storage_dir
    env["PYTHONPATH"] = os.getcwd() # Ensure we can import 'wheeler'

    # 1. Store
    print("1. Testing 'store' command...")
    run_command(["store", "Phase 4 CLI Test"], env=env)
    
    # 2. Recall
    print("2. Testing 'recall' command...")
    res = run_command(["recall", "Phase 4"], env=env)
    if "Phase 4 CLI Test" in res.stdout:
        print("   - [OK] Recall found the stored memory.")
    else:
        print("   - [FAIL] Recall failed to find memory.")

    # 3. Viz Run (Generate GIF)
    print("3. Testing 'viz_run' command...")
    gif_path = "docs/images/evolution.gif"
    os.makedirs("docs/images", exist_ok=True)
    run_command(["viz-run", "Evolution", "--output", gif_path, "--steps", "10"], env=env)
    
    if os.path.exists(gif_path):
        size = os.path.getsize(gif_path)
        print(f"   - [OK] GIF generated at {gif_path} ({size} bytes).")
    else:
        print(f"   - [FAIL] GIF generation failed.")

    print("--- Done ---")

if __name__ == "__main__":
    verify()
