from collections import defaultdict


words = open("names.txt", "r").read().split()

# length = len(words)
# firstTen = words[:10]
# maxLen = max(len(l) for l in words)
# minLen = min(len(l) for l in words)

# print(f"length = {length}, firstTen = {firstTen}, max = {maxLen}, min = {minLen}")

myDict = defaultdict(int)
for word in words[:3]:
    word = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(word, word[1:]):
        myDict[(ch1, ch2)] += 1


