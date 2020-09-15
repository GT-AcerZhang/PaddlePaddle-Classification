import argparse
import os
from ppcls.modeling import architectures
import paddle.fluid as fluid
import paddle_serving_client.io as serving_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",            type=str,  default='ResNet50_vd')
    parser.add_argument("-p", "--pretrained_model", type=str,  default='output/ResNet50_vd/best_model_in_epoch_1/ppcls')
    parser.add_argument("-o", "--output_path",      type=str,  default='output/serving_model/')
    parser.add_argument("--class_dim",              type=int,  default=24)
    parser.add_argument("--img_size",               type=int,  default=224)

    return parser.parse_args()


def create_input(img_size=224):
    image = fluid.data(name='feed_image', shape=[None, 3, img_size, img_size], dtype='float32')
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

    model = architectures.__dict__[args.model]()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()

    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            image = create_input(args.img_size)
            out = create_model(args, model, image, class_dim=args.class_dim)

    infer_prog = infer_prog.clone(for_test=True)
    fluid.load(program=infer_prog, model_path=args.pretrained_model, executor=exe)

    model_path = os.path.join(args.output_path, "ppcls_model")
    conf_path = os.path.join(args.output_path, "ppcls_client_conf")
    serving_io.save_model(model_path, conf_path, {"image": image}, {"prediction": out}, infer_prog)


if __name__ == "__main__":
    main()
