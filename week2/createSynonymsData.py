import argparse
import os
from pathlib import Path
import fasttext


parser = argparse.ArgumentParser(description='Process top words.')
general = parser.add_argument_group("general")
general.add_argument("--model", default='workspace/datasets/fasttext/title_model.bin',  help="The title model to use")
general.add_argument("--min_threshold", default=0.75, type=float, help="The minimum threshold for neighbors similarity (default is 0.75).")
general.add_argument("--input", default='/workspace/datasets/fasttext/top_words.txt',  help="The file containing the top words")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="The file to output to")

args = parser.parse_args()
model_file = args.model
min_threshold = args.min_threshold
input_file = args.input
output_file = args.output

input_path = Path(input_file)
input_dir = input_path.parent
if os.path.isdir(input_dir) == False:
        os.mkdir(input_dir)

output_path = Path(output_file)
output_dir = output_path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

output_rows = []

if __name__ == '__main__':
    model = fasttext.load_model(str(model_file))

    with open(input_file, 'r') as f:
        for line in f:
            neighbors = model.get_nearest_neighbors(line.strip())

            min_threshold_neighbor = []

            for neighbor in neighbors:
                if neighbor[0] >= min_threshold:
                    min_threshold_neighbor.append(neighbor[1])

            if len(min_threshold_neighbor) > 0:
                output_rows.append(line.strip() + "," + ",".join(min_threshold_neighbor))
    
    with open(output_file, 'w') as output:
        for row in output_rows:
            output.write(row + "\n")
