from pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re


def bpe(path: str, max_vocab_size: int, special_tokens: list[str]):
    #chunk and work on 1 chunk only
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=4,
            split_special_token=b"<|endoftext|>"
        )

        # Work on chunk 1 only
        start, end = boundaries[0], boundaries[1]

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    token_map = Counter() #byte tupple : occurance
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    i = 0

    for section in chunk.split(special_tokens[0]):
        i += 1
        if i % 10000 == 0:
            print(f"Word: {i}")

        for same in re.finditer(PAT, section):
            word = same.group()
            byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            token_map[byte_tuple] += 1

    vocab_size = 256

    while vocab_size < max_vocab_size:
        pair_counter = Counter()
        for keys, values in token_map.items():
            for i in range(len(keys)-1):
                pair = (keys[i], keys[i+1])
                pair_counter[pair] += values
        
        max_pair = max(pair_counter, key=lambda pair : (pair_counter[pair], pair))

        new_token = max_pair[0] + max_pair[1]
        new_token_map = Counter()

        for keys, frequency in token_map.items():
            new_keys = []
            i = 0

            while i < len(keys):
                if i < len(keys) - 1 and (keys[i], keys[i+1]) == max_pair:
                    new_keys.append(new_token)
                    i += 2
                else:
                    new_keys.append(keys[i])
                    i += 1

            new_token_map[tuple(new_keys)] += frequency

        token_map = new_token_map

        vocab_size += 1
    return new_token_map
    


bpe_ret = bpe(
    "../data/TinyStoriesV2-GPT4-train.txt",
    1000,
    ["<|endoftext|>"]
)

print(bpe_ret)