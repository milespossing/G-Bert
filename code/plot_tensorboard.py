from tensorboard import program as tb_program
tracking_address = '../saved'  # the path of your log file.

if __name__ == '__main__':

    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")