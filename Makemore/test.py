from collections import defaultdict
import torch

words = open("names.txt", "r").read().split()

# length = len(words)
# firstTen = words[:10]
# maxLen = max(len(l) for l in words)
# minLen = min(len(l) for l in words)

# print(f"length = {length}, firstTen = {firstTen}, max = {maxLen}, min = {minLen}")


a = torch.zeros(27, 27, dtype=torch.int32)
charToInt = defaultdict(int)
intToChar = defaultdict(str)

alphabet = (set("".join(words)))
alphabet.add(".")
alphabet = sorted(list(alphabet))
# print(alphabet)

for i, char in enumerate(alphabet):
    charToInt[char] = i
    intToChar[i] = char
# print(charToInt)

for word in words:
    word = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(word, word[1:]):
        a[charToInt[ch1],charToInt[ch2]] += 1

# print(a[0])
p = a[0].float()
p = p / sum(a[0])
print(p)

# def final(start_char, max_length=20):
#     result = [start_char]
#     current_char = start_char

#     for _ in range(max_length):
#         in_idx = charToInt[current_char]

#         out_idx = a[in_idx].argmax().item()
#         out_char = intToChar[out_idx]

#         if out_char == "<E>":
#             break

#         result.append(out_char)
#         current_char = out_char

#     return "".join(result)

# for char in alphabet:
#     print(f"{char} : {final(char)}")

