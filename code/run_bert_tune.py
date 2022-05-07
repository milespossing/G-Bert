from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboard import program as tb_program
from experiment_manager import main as exp_main, collect_all_args


if __name__ == "__main__":
    config_str = '--shuffle_inds -trn -evl --tp --use_single --gpus 1 --log_every_n_steps 1 --tg_int 5 ' \
                 '--max_epochs 20 --learning_rate 1.0e-3 --lr_exp_decay=0.7 --output_dir ../saved/bert_tune'
    for num_hidden_layers in [2, 3]:
        for num_attention_heads in [3, 4, 5]:
            args = collect_all_args(config_str)
            args.num_hidden_layers = num_hidden_layers
            args.num_attention_heads = num_attention_heads
            args.exp_name = f'layers_{num_hidden_layers}_heads_{num_attention_heads}'
            exp_main(args)
            last_output_dir = args.output_dir

    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', last_output_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")
