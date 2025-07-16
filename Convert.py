"""
Module Name: Convert.py

Description:
    Applies background subtraction to a batch of .mp4 videos in parallel, using either MOG2 or KNN.
    Original videos are first moved into a backup directory, then each is processed frame‑by‑frame
    to highlight moving objects and suppress static backgrounds. The processed videos overwrite
    the originals in the working directory.

Usage:
    python Convert.py \
        --path <video_directory> \
        --dest-dir <backup_directory> \
        [--max-workers <num_processes>] \
        [--subtractor <MOG2|KNN>]

Arguments:
    --path
        Directory containing the original .mp4 videos. (default: ".")
    --dest-dir
        Directory to move originals before processing; renamed with “_old” suffix if it exists, then recreated. (default: "unsubtracted_videos")
    --max-workers
        Number of parallel processes for conversion. (default: 10)
    --subtractor
        Background subtraction algorithm: "MOG2" or "KNN". (default: "MOG2")

Workflow:
    1. Enable multiprocessing support and configure logging.
    2. Change working directory to `--path`.
    3. Rename existing `--dest-dir` to `<dest-dir>_old` if present, then create a fresh `--dest-dir`.
    4. Move all `.mp4` files into `--dest-dir`.
    5. Use a ProcessPoolExecutor with `--max-workers` to run `convert_video` on each file:
         - Open video with OpenCV.
         - Instantiate the chosen subtractor.
         - For each frame: apply subtraction mask, write masked frame to a new .mp4 in the working dir.
         - Log progress every 10,000 frames.
    6. Release all resources and log completion.

Dependencies:
    - OpenCV (`cv2`) for video I/O and background subtraction.
    - `argparse`, `logging`, `os`, `subprocess`, `re` for CLI, filesystem, and logging.
    - `concurrent.futures`, `multiprocessing.freeze_support` for parallel processing.
"""


import cv2

import argparse
from multiprocessing import freeze_support
import os
import subprocess
import concurrent.futures
import re
import logging


def convert_video(subtract_type, file, old_video_repository):
    logging.info(f"Starting the conversion of the video {file}")
    try:
        cap = cv2.VideoCapture(os.path.join(old_video_repository, file))

        if subtract_type == "MOG2":
            subtractor = cv2.createBackgroundSubtractorMOG2()
        elif subtract_type == "KNN":
            subtractor = cv2.createBackgroundSubtractorKNN()
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(
            file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(cap.get(cv2.CAP_PROP_FPS)),
            (width, height),
        )
        logging.info(
            f"Starting the reading of the video {file}, with height {height} and width {width}"
        )

        count = 0
        while True:
            # masking done on a frame by frame basis
            ret, frame = cap.read()
            if not ret:
                break
            if count % 10000 == 0 and count != 0:
                logging.info(f"Processing frame {count} of {file}")
            fgMask = subtractor.apply(frame)
            masked = cv2.bitwise_and(frame, frame, mask=fgMask)
            count += 1
            writer.write(masked)
    except Exception as e:
        logging.error(f"Error processing the video {file} with error {e}")
    finally:
        cap.release()
        writer.release()
        logging.info(f"Reseased captures for video {file}")

 
if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Add background subtraction to videos")
    parser.add_argument(
        "--path", help="the path to the video files", required=False, type=str, default="."
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        required=False,
        help="the directory to move the old videos too",
        default="unsubtracted_videos",
    )
    parser.add_argument(
        "--max-workers",
        help="the number of workers to use in processing then videos",
        default=10,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--subtractor",
        help="the background subtractor to use",
        default="MOG2",
        type=str,
        required=False,
        choices=["MOG2", "KNN"],
    )
    args = parser.parse_args()

    os.chdir(args.path)
    file_list = os.listdir()
    if args.dest_dir in file_list:
        subprocess.run(f"mv {args.dest_dir} {args.dest_dir}_old", shell=True)

    os.mkdir(args.dest_dir)

    command = f"mv *.mp4 {args.dest_dir}"
    subprocess.run(command, shell=True)
    old_dir_list = os.listdir(args.dest_dir)
    file_list = list(set([file for file in file_list if re.search(r".mp4$", file)]))

    logging.info(
        f"Finished the moving of old videos to the destination directory, creating subtractor, file list is {file_list}"
    )

    logging.info("Starting the conversion of the videos")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(convert_video, args.subtractor, file, args.dest_dir)
            for file in file_list
        ]
        concurrent.futures.wait(futures)
    logging.info("Finished the conversion of the videos")
