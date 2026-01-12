class MemoryReader:
    def __init__(self, brain):
        self.brain = brain

    def read_state(self):
        try:
            return self.brain.state
        except AttributeError as e:
            print(f"Error reading state: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Simulating a brain object with a state attribute
    class Brain:
        def __init__(self):
            self.state = "active"

    brain_instance = Brain()
    reader = MemoryReader(brain_instance)
    print(reader.read_state())
