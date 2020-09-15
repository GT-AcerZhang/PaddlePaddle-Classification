import argparse
import numpy as np
from ppcls.utils import logger
from ppcls.data import Reader
from ppcls import get_config
from ppcls.modeling import architectures
import paddle.fluid as fluid
import paddleslim as slim

import program


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("--class_dim", type=int, default=23)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet50_vd.yaml',
        help='config file path')
    parser.add_argument(
        '--use_quant',
        type=bool,
        default=True,
        help='If use slim quant train.')

    return parser.parse_args()


def create_input(img_size=224):
    image = fluid.data(name='image', shape=[None, 3, img_size, img_size], dtype='float32')
    return image


def create_model(args, model, input, class_dim=1000):
    if args.model == "GoogLeNet":
        out, _, _ = model.net(input=input, class_dim=class_dim)
    else:
        out = model.net(input=input, class_dim=class_dim)
        out = fluid.layers.softmax(out)
    return out


def main():
    args = parse_args()
    config = get_config(args.config, overrides=args.override, show=True)

    model = architectures.__dict__[args.model]()

    use_gpu = config.get("use_gpu", True)
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()
    exe = fluid.Executor(places[0])

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    if not config.get('use_ema'):
        train_dataloader, train_fetchs = program.build(config,
                                                       train_prog,
                                                       startup_prog,
                                                       is_train=True,
                                                       is_distributed=False)
    else:
        train_dataloader, train_fetchs, ema = program.build(config,
                                                            train_prog,
                                                            startup_prog,
                                                            is_train=True,
                                                            is_distributed=False)

    valid_prog = fluid.Program()
    valid_dataloader, valid_fetchs = program.build(config,
                                                   valid_prog,
                                                   startup_prog,
                                                   is_train=False,
                                                   is_distributed=False)
    # clone to prune some content which is irrelevant in valid_prog
    valid_prog = valid_prog.clone(for_test=True)

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            image = create_input(args.img_size)
            out = create_model(args, model, image, class_dim=args.class_dim)

    infer_prog = train_prog.clone(for_test=True)
    fluid.load(program=train_prog, model_path=args.pretrained_model, executor=exe)

    fluid.io.save_inference_model(dirname=args.output_path,
                                  feeded_var_names=[image.name],
                                  main_program=infer_prog,
                                  target_vars=out,
                                  executor=exe,
                                  model_filename='model',
                                  params_filename='params')

    train_reader = Reader(config, 'train')()
    train_dataloader.set_sample_list_generator(train_reader, places)

    if config.validate:
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, places)

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
        val_quant_program = slim.quant.quant_aware(valid_prog, exe.place, for_test=True)

        # for idx, batch in enumerate(valid_dataloader()):
        #     metrics = exe.run(program=val_quant_program, feed=batch, fetch_list=fetch_list)
        #     for i, m in enumerate(metrics):
        #         metric_list[i].update(np.mean(m), len(batch[0]))
        #     fetchs_str = ''.join([str(m.value) + ' ' for m in metric_list])
        #
        #     if idx % 10 == 0:
        #         logger.info("quant valid: " + fetchs_str)

        # save inference model
        float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
        fluid.io.save_inference_model(dirname='./inference_model/float',
                                      feeded_var_names=[image.name],
                                      target_vars=out,
                                      executor=exe,
                                      main_program=float_prog)
        fluid.io.save_inference_model(dirname='./inference_model/int8',
                                      feeded_var_names=[image.name],
                                      target_vars=out,
                                      executor=exe,
                                      main_program=int8_prog)


if __name__ == "__main__":
    main()
