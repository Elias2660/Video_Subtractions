import cv2
import argparse
from multiprocessing import freeze_support
import os
import subprocess
import concurrent
import re
import logging


def convert_video(subtractor, file, old_video_repository): 
    try:
        cap = cv2.VideoCapture(os.path.join(old_video_repository, file))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(
            file,
            cv2.VideoWriter_fourcc(*"DIVX"),
            int(cap.get(cv2.CAP_PROP_FPS)),
            (width, height),
        )
        logging.info(f"Starting the conversion of the video {file}")
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % 1000 == 0:
                logging.info(f"Processing frame {count} of {file}")
            fgMask = subtractor.apply(frame)
            masked = cv2.bitwise_and(frame, frame, mask=fgMask)


            writer.write(masked)
    except Exception as e:
        logging.error(f"Error processing the video {file} with error {e}")
    finally:
        cap.release()
        writer.release()
        logging.info(f"Reseased captures for video {file}")

if __name__ == "__main__":
    freeze_support()
    
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    
    parser = argparse.ArgumentParser(description="Add background subtraction to videos")
    parser.add_argument("--path", help="the path to the video", default=".")
    parser.add_argument(
        "--dest-dir",
        help="the directory to move the old videos too",
        default="unsubtracted_videos",
    )
    parser.add_argument(
        "--max-workers",
        help="the number of workers to use in processing then videos",
        default=10,
    )
    parser.add_argument(
        "--subtractor",
        help="the background subtractor to use",
        default="MOG2",
        choices=["MOG2", "KNN"],
    )
    args = parser.parse_args()

    os.chdir(args.path)
    file_list = os.listdir()
    if args.dest_dir in file_list:
        subprocess.run(f"mv {args.dest_dir} {args.dest_dir}_old", shell=True)

    os.mkdir(args.dest_dir)

    command = f"mv *.mp4 {args.dest_dir}"
    
    old_dir_list = os.listdir(args.dest_dir)
    file_list = list(set([file for file in file_list if re.search(r".mp4$", file)]))

    logging.info("Finished the moving of old videos to the destination directory, creating subtractor")
    if args.subtractor == "MOG2":
        subtractor = cv2.createBackgroundSubtractorMOG2()
    elif args.subtractor == "KNN":
        subtractor = cv2.createBackgroundSubtractorKNN()
        
    logging.info("Starting the conversion of the videos")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        for file in file_list:
            if file.endswith(".mp4"):
                futures = [
                    executor.submit(convert_video, subtractor, file, args.dest_dir) for file in file_list
                ]
