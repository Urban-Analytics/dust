# Imports
import pickle

# Main
# x = {'A': [1, 2, 3, 4, 5],
#      'B': [2, 3, 4, 5, 6]}

# with open('./results/models/test.pkl', 'wb') as f:
#     pickle.dump(x, f)

with open('./results/models/test.pkl', 'rb') as f:
    x = pickle.load(f)

print(x)
