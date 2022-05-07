from tensorboard import program as tb_program
# tracking_address = '../saved/05_06_session/ablation_take2'
# tracking_address = '../saved/lr_tune_take2'
# tracking_address = '../saved/bert_tune_take2'
tracking_address = '../saved/05_06_session/seed'
if __name__ == '__main__':
    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")