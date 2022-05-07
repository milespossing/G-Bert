from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboard import program as tb_program
from experiment_manager import main as exp_main, collect_all_args


if __name__ == "__main__":
    with open('batch_config.txt') as f:
        runs_config = f.readlines()
    last_output_dir = None
    for i_config, config_str in enumerate(runs_config):
        args = collect_all_args(config_str)
        exp_main(args)
        last_output_dir = args.output_dir

    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', last_output_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")
