import os
import sys


valid_char="ABCDEFGHIJKLMNOPQRSTUVWXYZ'"

def is_valid(w):
    for c in w:
        if c not in valid_char:
            return False
    return True


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input_file output_file", file=sys.stderr)
        os.exit(1)

    print(len(valid_char))

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    words = set()
    with open(input_file) as f:
        for line in f:
            for word in line.split():
                words.add(word.upper())

    words = [w for w in words if is_valid(w)]

    with open(output_file, 'w') as f:
        for word in sorted(words):
            s = word + '\t'
            for c in word:
                s += f"{c} "
            s += '|'
            print(s, file=f)

if __name__ == "__main__":
    main()