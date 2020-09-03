#!/usr/bin/env python3


from argparse import ArgumentParser
import os
import re

import pdf2image


argp = ArgumentParser()
argp.add_argument("--dpi", type=int, default=200)
argp.add_argument("source_dir")
argp.add_argument("dest_dir")
args = argp.parse_args()

os.makedirs(args.dest_dir, exist_ok=True)

for file_name in os.listdir(args.source_dir):
    output_name = re.sub("\s+", "_", os.path.splitext(file_name)[0]).lower()
    print("Converting '%s' to '%s'..." % (file_name, output_name))
    output_base = os.path.join(args.dest_dir, output_name + "_page_%d_%d.png")

    imgs = pdf2image.convert_from_path(
         os.path.join(args.source_dir, file_name),
         dpi=args.dpi,
         fmt="png")

    for i, img in enumerate(imgs):
        img = img.convert("L").convert("RGB")
        img.save(output_base % (i, 0))
        for j in range(1, 4):
            img = img.rotate(90, expand=True)
            img.save(output_base % (i, j))
