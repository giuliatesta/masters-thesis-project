import pickle

# Path to the pickled file
file_path = './results/trial_0_LPStates.pickled'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)

# Now you can use the data
for d in data:
    print(f"!!!! {d}")
    print(d)
