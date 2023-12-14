import os
import sys
import argparse
import functools
from glob import glob
from pathlib import Path

import torch
import numpy as np
import monai.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import utils
import vision_transformer as vits


def main(args):

    # define the model and load pretrained weights
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](
            in_chans=3,
            patch_size=args.patch_size,
            num_classes=0)
        embed_dim = model.embed_dim
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # define image paths and labels
    image_pths = glob(os.path.join(args.data_path, '**', '*.nii.gz'))
    labels = [path.split(os.sep)[-2] for path in image_pths]

    # define preprocessing transforms
    transform = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.Orientation(axcodes='RA'),
        transforms.ScaleIntensityRange(a_min=-100, a_max=300, b_min=0, b_max=1, clip=True),
        transforms.ToTensor(),
    ])

    # loop through image, perform forward-pass and save features
    features = torch.zeros([len(image_pths), embed_dim])
    with torch.no_grad():
        for idx, image_pth in enumerate(image_pths):
            if torch.cuda.is_available():
                image = transform(image_pth).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
            else:
                image = transform(image_pth).unsqueeze(0).repeat(1, 3, 1, 1)
            out = model(image).cpu()  # model.forward_features(image)['x_norm_clstoken'].cpu()

            features[idx, :] = out

    # make t-sne plot
    t_sne = TSNE(n_components=2, perplexity=5, random_state=0).fit_transform(features.numpy())
    colors = ['b', 'g', 'r', 'y', 'k']
    unique_labels = functools.reduce(lambda re, x: re+[x] if x not in re else re, labels, [])
    for i, label in enumerate(labels):
        for j, unique in enumerate(unique_labels):
            if label == unique:
                plt.scatter(t_sne[i, 0], t_sne[i, 1], marker='.', color=colors[j], s=50)
                continue
    plt.savefig(os.path.join(args.output_dir, 't_sne.png'))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(t_sne, np.array(labels))
    y_pred = knn.predict(t_sne)
    print("Accuracy 5-NN classifier: ", accuracy_score(np.array(labels), y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Training and plotting of t-SNE plot with KNN classifier on fine-tuned
        classification set""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pretrained weights of encoder.')
    parser.add_argument('--checkpoint_key', default='teacher', type=str, help="""Key to use in the checkpoint""")
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes in the dataset')
    parser.add_argument('--data_path', default='/path/to/data/', type=str)
    parser.add_argument('--output_dir', default="./", help='Path to save t-sne plot')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

