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
        "--path", help="the path to the video", required=False, type=str, default="."
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
