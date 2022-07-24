import sys
import time
from calc_length import read_map_and_calc_length
from tr_tensor import train_model, check_picture


def exit():
    print("Exit...")
    sys.exit()


actions = {
    "length_task": read_map_and_calc_length,
    "train_model": train_model,
    "check_picture": check_picture,
    "exit": exit
}


def print_parts(parts):
    for i in range(len(parts)):
        print(f"{i}:", f"{parts[i]}")


def main():
    list_key = list(actions.keys())
    print_parts(list_key)
    option = int(input("Choose option: "))
    start = time.time()
    actions[list_key[option]]()
    print("Finish:", time.time() - start)


if __name__ == '__main__':
    while True:
        main()
