from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboard import program as tb_program
from experiment_manager import main as exp_main, collect_all_args


if __name__ == "__main__":
    config_str = '--shuffle_inds -trn -evl -tp --use_single --gpus 1 --log_every_n_steps 1 -tg_int 5 ' \
                 '--max_epochs 30 --output_dir ../saved/lr_tune'
    for lr in [5e-3, 1e-3, 5e-4, 1e-4]:
        for decay_fact in [1., .9, .8, .7]:
            args = collect_all_args(config_str)
            args.learning_rate = lr
            args.lr_exp_decay = decay_fact
            args.exp_name = f'lr_{lr}_fact_{decay_fact}'
            exp_main(args)
            last_output_dir = args.output_dir

    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', last_output_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")
