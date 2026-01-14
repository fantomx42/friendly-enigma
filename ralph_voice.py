import sys
import os
import time

# Ensure ralph_core is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ralph_core.voice import voice
from ralph_core.triggers.queue_manager import queue

def main():
    print("🎙️  Ralph Voice Interface Online")
    voice.speak("Ralph Voice Interface Online. Waiting for command.")
    
    while True:
        # Listen Loop
        text = voice.listen(duration=5)
        
        if not text:
            continue
            
        text_lower = text.lower()
        
        # 1. Status Report
        if "status report" in text_lower or "status check" in text_lower:
            voice.speak("Checking system status.")
            
            # Read STATUS.md
            if os.path.exists("STATUS.md"):
                with open("STATUS.md", 'r') as f:
                    # Just read the first few lines to avoid reading whole file
                    content = f.read()[:200] 
                    voice.speak(f"Here is the summary: {content}")
            else:
                voice.speak("No status file found.")

        # 2. Task Queuing
        elif "ralph" in text_lower:
            # "Hey Ralph, build a website" -> "build a website"
            # Split by 'ralph'
            parts = text_lower.split("ralph", 1)
            if len(parts) > 1:
                task = parts[1].strip()
                if task:
                    queue.add_task("Voice", task)
                    voice.speak(f"I have queued the task: {task}")
                else:
                    voice.speak("Yes? I am listening.")
        
        # 3. Exit
        elif "shutdown" in text_lower or "exit" in text_lower:
            voice.speak("Shutting down voice interface.")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()
