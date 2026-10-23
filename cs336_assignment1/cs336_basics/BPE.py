from pretokenization_example import find_chunk_boundaries

def bpe(path: str, vocab_size: int, special_tokens: list[str]):

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

    


bpe(
    "../data/TinyStoriesV2-GPT4-train.txt",
    1000,
    ["<|endoftext|>"]
)