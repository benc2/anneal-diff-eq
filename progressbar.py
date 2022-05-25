import sys


class ProgressBar:
    def __init__(self, ticks, width=20) -> None:
        self.ticks = ticks
        self.width = width
        self.current_tick = 0
        self.draw()

    def draw(self, extra=""):
        done_width = round((self.current_tick) / self.ticks * self.width)
        pb = (
            done_width * "="
            + ">"
            + (self.width - done_width - 1) * "-"
            + f" {round(self.current_tick/self.ticks*100)}% "
            + extra
        )
        # sys.stdout.write("\b" + pb)
        # sys.stdout.flush()
        print("\r" + pb, end="")

    def tick(self, extra=""):
        self.current_tick += 1
        self.draw(extra)


if __name__ == "__main__":
    import time

    pb = ProgressBar(20)
    for i in range(20):
        pb.tick(f"({i}/20)")
        time.sleep(0.1)
