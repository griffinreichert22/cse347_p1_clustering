# testing recieving user input from keyboard

## Get user input from command line
print("Welcome to my clustering alg!")
print("  [1] K-means clustering")
print("  [2] Spectral clustering")
alg = input("Please enter the number of the algorithm you wish to use: ")
while alg != "1" and alg != "2":
    alg = input("Whoops! Please only enter '1' or '2' ('q' to quit): ")
    if alg == "q":
        break
## Read in data from text files
print("reading in data")
# file paths
data_fp = "../data/"
cho_fp = data_fp + "cho.txt"
iyer_fp = data_fp + "iyer.txt"

cho_file = open(cho_fp, "r")
cho_data = cho_file.read()
# print(cho_data)
iyer_file = open(iyer_fp, "r")
iyer_data = iyer_file.read()
# print(iyer_data)

if alg == "1":
    print("Do K-means here")
elif alg == "2":
    print("Do spectral here")