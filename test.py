from string import punctuation


def normalise_file():
    FILENAME = "shakespeare.txt"

    with open(FILENAME) as f:
        data = f.read()

    data = data.lower()

    for p in punctuation:
        data = data.replace(p, "")

    data = data.replace("\n", " ")
    data = data.replace("\"", "")
    for _ in range(10):
        data = data.replace("  ", " ")

    with open(FILENAME, "w") as f:
        f.write(data)


# normalise_file()


with open("datasets/dracula.txt", encoding="utf8") as f:
    chars = list(f.read())
    print(*chars[:1000])
