# Video Background Subtraction

A command‑line tool for applying background subtraction (MOG2 or KNN) to batches of `.mp4` videos in parallel.  
Original videos are moved to a backup directory before processing, then each video is processed frame‑by‑frame to highlight moving objects and suppress the static background.

This project is most used with the [Unified‑bee‑Runner](https://github.com/Elias2660/Unified-bee-Runner).

---

## Requirements

- Python 3.12
- OpenCV (`cv2`)
- NumPy

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment

   ```
   python3 -m venv venv
   source venv/bin/activate    # On Windows: `venv\Scripts\activate`
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python Convert.py \
  --path <video_directory> \
  --dest-dir <backup_directory> \
  [--max-workers <num_processes>] \
  [--subtractor <MOG2|KNN>]
```

- `--path` (default: `.`) \
  Directory containing the original .mp4 videos.

- `--dest-dir` (default: `unsubtracted_videos`) \
  Directory to which originals are moved. If it already exists, it will be renamed to `<dest-dir>_old`.

- `--max-workers` (default: `10`) \
  Number of parallel processes to use for conversion.

- `--subtractor` (default: `MOG2`) \
  Background subtraction algorithm: `MOG2` or `KNN`.

## Example

```bash
python Convert.py \
  --path ./videos \
  --dest-dir ./processed_backup \
  --max-workers 5 \
  --subtractor KNN
```

## Logging

Logs are printed to the console at INFO level with timestamps. They include:

- Start and finish of each video conversion

- Progress updates every 10,000 frames

- Any errors or exceptions encountered

## License

This project is licensed under the (MIT License)[LICENSE].
