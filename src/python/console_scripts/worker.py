import sys

import marius as m


def main():
    print(sys.argv)
    m.worker(len(sys.argv), sys.argv)


if __name__ == "__main__":
    sys.exit(main())
