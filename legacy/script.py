# Example script to demonstrate the correction

# Define the state dictionary with 'brain' as a key
state = {
    'brain': 'active'
}

def check_brain_status():
    # Correctly access 'brain' from the state dictionary
    if state['brain'] == 'active':
        print("Brain is active.")
    else:
        print("Brain is not active.")

# Call the function to test
check_brain_status()
