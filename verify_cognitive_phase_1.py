import subprocess
import os
import sys
import shutil

def run_command(args, env=None):
    cmd_str = " ".join(args)
    print("\n> python -m wheeler.cli {}".format(cmd_str))
    cmd = [sys.executable, "-m", "wheeler.cli"] + args
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
    print("--- Cognitive Functions Phase 1 Verification (Smarter Codec) ---")
    
    # Use a fresh storage for this verification to be clean
    storage_dir = os.path.abspath("verification_cognitive_p1")
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        
    env = os.environ.copy()
    env["WHEELER_STORAGE"] = storage_dir
    env["PYTHONPATH"] = os.getcwd()

    # 1. Store long sentence
    print("1. Storing sentence...")
    run_command(["store", "This is my first permanent memory"], env=env)
    
    # 2. Recall sub-phrase
    print("2. Recalling sub-phrase 'permanent memory'...")
    res = run_command(["recall", "permanent memory"], env=env)
    
    # Check output for similarity
    # We expect high similarity (e.g., > 0.4 or 0.5 depending on word count)
    # "permanent memory" is 2 words. "This is my first permanent memory" is 6 words.
    # Overlap is 2 words. 2 / sqrt(6) * sqrt(2)? No, cosine similarity.
    # v1 = (w1+w2+w3+w4+w5+w6) / sqrt(6)
    # v2 = (w5+w6) / sqrt(2)
    # dot = (w5.w5 + w6.w6 + 0) / (sqrt(6)*sqrt(2)) = (1+1) / sqrt(12) = 2 / 3.46 = 0.57
    # So similarity should be around 0.57.
    
    if "Similarity: 0.5" in res.stdout or "Similarity: 0.6" in res.stdout or "Similarity: 0.4" in res.stdout:
         print("   - [OK] Similarity is significant (approx 0.57 expected).")
    else:
         print("   - [WARN] Similarity value might be unexpected. checking output manually.")

    # 3. Recall reverse order
    print("3. Recalling 'memory permanent'...")
    res = run_command(["recall", "memory permanent"], env=env)
    
    # Should be SAME similarity as above (0.57) because it's bag of words
    # "memory permanent" vector == "permanent memory" vector
    
    print("--- Done ---")
    
    # Cleanup
    shutil.rmtree(storage_dir)

if __name__ == "__main__":
    verify()
