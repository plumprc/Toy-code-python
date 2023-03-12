import imageio
from pathlib import Path
from IPython.display import Image
import argparse

def generate_gif(path):
    images = []
    for file in sorted([file for file in Path(path).glob('*.jpg')]):
        images.append(imageio.imread(file))
    
    imageio.mimsave('recon_image.gif', images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='img', help='path of images')
    args = parser.parse_args()
    generate_gif(args.path)
