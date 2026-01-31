import sys
import os

# Ensure we can find the ralph_core package
# We assume this script is running from ai_tech_stack/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ralph_core.wheeler import WheelerMemoryBridge
except ImportError as e:
    # Fallback for when running from root or different structure
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        from ai_tech_stack.ralph_core.wheeler import WheelerMemoryBridge
    except ImportError:
        print(f"Error importing WheelerMemoryBridge: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        # If no prompt provided, just return empty
        return
    
    prompt = " ".join(sys.argv[1:])
    
    try:
        bridge = WheelerMemoryBridge()
        context = bridge.recall(prompt)
        if context:
            print(context)
        
        # Shutdown to ensure threads clean up if any
        bridge.shutdown()
    except Exception as e:
        print(f"Error in wheeler_recall: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
