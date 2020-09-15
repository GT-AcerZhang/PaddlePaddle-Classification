import argparse
import os
import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter
from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model, save_model
from ppcls.utils import logger
import program
import paddleslim as slim


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet50_vd.yaml',
        help='config file path')
    parser.add_argument(
        '--vdl_dir',
        type=str,
        default="logs",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '--use_quant',
        type=bool,
        default=True,
        help='If use slim quant train.')
    args = parser.parse_args()
    return args


def main(args):
    config = get_config(args.config, overrides=args.override, show=True)
    # assign the place
    use_gpu = config.get("use_gpu", True)
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    best_top1_acc = 0.0  # best top1 acc record

    if not config.get('use_ema'):
        train_dataloader, train_fetchs, out = program.build(config,
                                                            train_prog,
                                                            startup_prog,
                                                            is_train=True,
                                                            is_distributed=False)
    else:
        train_dataloader, train_fetchs, ema, out = program.build(config,
                                                                 train_prog,
                                                                 startup_prog,
                                                                 is_train=True,
                                                                 is_distributed=False)

    if config.validate:
        valid_prog = fluid.Program()
        valid_dataloader, valid_fetchs, _ = program.build(config,
                                                          valid_prog,
                                                          startup_prog,
                                                          is_train=False,
                                                          is_distributed=False)
        # clone to prune some content which is irrelevant in valid_prog
        valid_prog = valid_prog.clone(for_test=True)

    # create the "Executor" with the statement of which place
    exe = fluid.Executor(places[0])
    # Parameter initialization
    exe.run(startup_prog)

    # load model from 1. checkpoint to resume training, 2. pretrained model to finetune
    init_model(config, train_prog, exe)

    train_reader = Reader(config, 'train')()
    train_dataloader.set_sample_list_generator(train_reader, places)

    compiled_train_prog = program.compile(config, train_prog, train_fetchs['loss'][0].name)

    if config.validate:
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, places)
        compiled_valid_prog = program.compile(config, valid_prog, share_prog=compiled_train_prog)

    vdl_writer = LogWriter(args.vdl_dir)

    for epoch_id in range(config.epochs):
        # 1. train with train dataset
        program.run(train_dataloader, exe, compiled_train_prog, train_fetchs, epoch_id, 'train', config, vdl_writer)

        # 2. validate with validate dataset
        if config.validate and epoch_id % config.valid_interval == 0:
            if config.get('use_ema'):
                logger.info(logger.coloring("EMA validate start..."))
                with ema.apply(exe):
                    _ = program.run(valid_dataloader, exe,
                                    compiled_valid_prog, valid_fetchs,
                                    epoch_id, 'valid', config)
                logger.info(logger.coloring("EMA validate over!"))

            top1_acc = program.run(valid_dataloader, exe, compiled_valid_prog, valid_fetchs, epoch_id, 'valid', config)
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                message = "The best top1 acc {:.5f}, in epoch: {:d}".format(best_top1_acc, epoch_id)
                logger.info("{:s}".format(logger.coloring(message, "RED")))
                if epoch_id % config.save_interval == 0:
                    model_path = os.path.join(config.model_save_dir, config.ARCHITECTURE["name"])
                    save_model(train_prog, model_path, "best_model_in_epoch_" + str(epoch_id))

        # 3. save the persistable model
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir, config.ARCHITECTURE["name"])
            save_model(train_prog, model_path, epoch_id)

        # quant train
        if args.use_quant:
            # train
            quant_program = slim.quant.quant_aware(train_prog, exe.place, for_test=False)

            fetch_list = [f[0] for f in train_fetchs.values()]
            metric_list = [f[1] for f in train_fetchs.values()]
            for idx, batch in enumerate(train_dataloader()):
                metrics = exe.run(program=quant_program, feed=batch, fetch_list=fetch_list)
                for i, m in enumerate(metrics):
                    metric_list[i].update(np.mean(m), len(batch[0]))
                fetchs_str = ''.join([str(m.value) + ' ' for m in metric_list])

                if idx % 10 == 0:
                    logger.info("quant train : " + fetchs_str)

            # eval
            val_quant_program = slim.quant.quant_aware(train_prog, exe.place, for_test=True)

            # for idx, batch in enumerate(valid_dataloader()):
            #     metrics = exe.run(program=val_quant_program, feed=batch, fetch_list=fetch_list)
            #     for i, m in enumerate(metrics):
            #         metric_list[i].update(np.mean(m), len(batch[0]))
            #     fetchs_str = ''.join([str(m.value) + ' ' for m in metric_list])
            #
            #     if idx % 10 == 0:
            #         logger.info("quant valid: " + fetchs_str)

            # save inference model
            # image = create_input(img_size=224)
            # out = create_model(os.path.basename(args.config)[:-5], image, class_dim=24)

            float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
            fluid.io.save_inference_model(dirname='./inference_model/float',
                                          feeded_var_names=['feed_image'],
                                          target_vars=out,
                                          executor=exe,
                                          main_program=float_prog)
            fluid.io.save_inference_model(dirname='./inference_model/int8',
                                          feeded_var_names=['feed_image'],
                                          target_vars=out,
                                          executor=exe,
                                          main_program=int8_prog)


if __name__ == '__main__':
    args = parse_args()
    main(args)
