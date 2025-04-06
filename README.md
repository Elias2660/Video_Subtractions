# Video Background Subtraction

This project provides a script to perform background subtraction on videos using OpenCV. The script processes videos in a specified directory and saves the processed videos with the background subtracted.

This project is most used with the [Unified-bee-Runner](https://github.com/Elias2660/Unified-bee-Runner).

## Requirements
 
- Python 3.12
- OpenCV
- NumPy

## Installation

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```sh
python Convert.py --path <path-to-videos> --dest-dir <destination-directory> --max-workers <number-of-workers> --subtractor <subtractor-type>
```

## Arguments

`--path`: The path to the directory containing the videos. Default is the current directory. 

`--dest-dir`: The directory to move the old videos to. Default is unsubtracted_videos. 

`--max-workers`: The number of workers to use for processing the videos. Default is 10. 

`--subtractor`: The background subtractor to use. Choices are MOG2 and KNN. Default is MOG2.

## Example

```sh
python Convert.py --path ./videos --dest-dir ./processed_videos --max-workers 5 --subtractor KNN
```

## Logging

The script logs its progress and any errors encountered during processing. Logs are printed to the console with timestamps.

## License

This project is licensed under the [MIT License](LICENSE).
