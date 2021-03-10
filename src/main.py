# testing recieving user input from keyboard

print("Welcome to my clustering alg!")
print("  [1] K-means clustering")
print("  [2] Spectral clustering")
alg = input("Please enter the number of the algorithm you wish to use: ")
while alg != "1" and alg != "2":
    alg = input("Whoops! Please only enter '1' or '2' ('q' to quit): ")
    if alg == "q":
        break
print("reading in data")

if alg == "1":
    print("Do K-means here")
elif alg == "2":
    print("Do spectral here")