import sys
import os
import argparse

# Ensure we can find the ralph_core package
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
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File containing output to store")
    parser.add_argument("--type", help="Type of memory", default="iteration")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(args.file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        bridge = WheelerMemoryBridge()
        # We store the content
        bridge.remember(content)
        
        # Shutdown
        bridge.shutdown()
    except Exception as e:
        print(f"Error in wheeler_store: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
