import os
import time
import json
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import chromadb
from chromadb.utils import embedding_functions

# OpenVINO might not be available yet if just installed, 
# so we wrap the import to allow the script to be saved and then initialized.
try:
    import openvino_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Paths
BRAIN_DIR = "/home/tristan/ralph_brain"
HOT_DIR = os.path.join(BRAIN_DIR, "hot")
COLD_DIR = os.path.join(BRAIN_DIR, "cold")
SUMMARIES_DIR = os.path.join(COLD_DIR, "summaries")
ARCHIVE_DIR = os.path.join(COLD_DIR, "archive")
MODEL_DIR = os.path.join(BRAIN_DIR, "models/librarian")

# Thresholds
HOT_FILE_MAX_BYTES = 4096  # 4KB as per plan

logging.basicConfig(level=logging.INFO, format='[Librarian] %(message)s')

class LibrarianHandler(FileSystemEventHandler):
    def __init__(self, librarian):
        self.librarian = librarian

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self.librarian.check_file(event.src_path)

class Librarian:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(BRAIN_DIR, "vector_db"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="ralph_memory",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self.pipe = None
        if OPENVINO_AVAILABLE and os.path.exists(MODEL_DIR):
            try:
                # Assuming Phi-3 INT4 OpenVINO model
                # Use 'AUTO' to let OpenVINO decide best hardware (NPU/GPU/CPU)
                self.pipe = openvino_genai.LLMPipeline(MODEL_DIR, "AUTO")
                logging.info("OpenVINO Pipeline initialized (AUTO).")
            except Exception as e:
                logging.warning(f"Failed to initialize OpenVINO pipeline: {e}. Falling back to CPU/Dummy.")

    def check_file(self, file_path):
        try:
            size = os.path.getsize(file_path)
            if size > HOT_FILE_MAX_BYTES:
                logging.info(f"Threshold exceeded for {os.path.basename(file_path)} ({size} bytes). Migrating...")
                self.migrate_context(file_path)
        except OSError:
            pass

    def migrate_context(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 10: return

        # Split: Oldest 50% to move, Newest 50% stays
        mid = len(lines) // 2
        to_migrate = "".join(lines[:mid])
        to_keep = "".join(lines[mid:])

        filename = os.path.basename(file_path)
        timestamp = int(time.time())

        # 1. Summarize (NPU)
        summary = self.summarize(to_migrate)
        
        # 2. Save Summary (Cold Storage)
        summary_path = os.path.join(SUMMARIES_DIR, f"summary_{timestamp}_{filename}")
        with open(summary_path, 'w') as f:
            f.write(summary)

        # 3. Save Raw Text (Archive)
        archive_path = os.path.join(ARCHIVE_DIR, f"archive_{timestamp}_{filename}")
        with open(archive_path, 'w') as f:
            f.write(to_migrate)

        # 4. Vectorize
        self.collection.add(
            documents=[to_migrate],
            metadatas=[{"source": filename, "type": "archive", "timestamp": timestamp}],
            ids=[f"doc_{timestamp}"]
        )

        # 5. Truncate Hot File
        with open(file_path, 'w') as f:
            f.write(to_keep)
        
        logging.info(f"Migration complete: {len(to_migrate)} characters archived.")

    def summarize(self, text):
        if self.pipe:
            prompt = f"Summarize the following code/task notes concisely:\n\n{text}\n\nSummary:"
            return self.pipe.generate(prompt, max_new_tokens=256)
        else:
            # Fallback/Dummy logic
            return f"[Draft Summary] Content starting with: {text[:100]}..."

    def run(self):
        observer = Observer()
        observer.schedule(LibrarianHandler(self), HOT_DIR, recursive=False)
        observer.start()
        logging.info(f"Monitoring {HOT_DIR} for context migration...")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    librarian = Librarian()
    librarian.run()
