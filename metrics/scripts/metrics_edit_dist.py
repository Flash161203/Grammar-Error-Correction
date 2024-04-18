import json
import editdistance
import os

current_directory = os.getcwd()

# list all available json data files
json_file_list = []

for filename in os.listdir(current_directory):
    if filename.endswith('.json') and os.path.isfile(os.path.join(current_directory, filename)):
        json_file_list.append(filename)

print(json_file_list)

# generate edit dist metrics for each predictions file
for predfile in json_file_list:

  with open(predfile, 'r') as file:
      data = json.load(file)

  inputs = data["inputs"]
  predictions = data["predictions"]

  # get Levenshtein edit distance between each pair
  distances = []
  for input_text, prediction in zip(inputs, predictions):
      distance = editdistance.eval(input_text, prediction)
      distances.append(distance)

  average_distance = sum(distances) / len(distances)

  metrics = {"average_edit_dist": average_distance, "edit_dists" : distances}

  prefix = predfile.split("_data.json")[0]
  result_file = f"{prefix}_edit_dist.txt"

  with open(result_file, "w") as f:
          f.write(json.dumps(metrics))

# generate edit dist metrics by cefr (A, B, C) for each predictions file
for predfile in json_file_list:
  with open(predfile, 'r') as file:
      data = json.load(file)

  # group elements by CEFR level
  grouped_data = {}

  for input_text, prediction, reference, id_, cefr in zip(data["inputs"], data["predictions"], data["references"], data["ids"], data["cefr"]):
      cefr = cefr[0]
      if cefr not in grouped_data:
          grouped_data[cefr] = {"inputs": [], "predictions": [], "references": [], "ids": []}
      grouped_data[cefr]["inputs"].append(input_text)
      grouped_data[cefr]["predictions"].append(prediction)
      grouped_data[cefr]["references"].append(reference)
      grouped_data[cefr]["ids"].append(id_)

  metrics = {}

  for cefr, data in grouped_data.items():
      inputs = data["inputs"]
      predictions = data["predictions"]
      distances = []

      for input_text, prediction in zip(inputs, predictions):
          distance = editdistance.eval(input_text, prediction)
          distances.append(distance)
      average_distance = sum(distances) / len(distances)
      metrics[cefr] = {"average_edit_dist": average_distance, "edit_dists" : distances}

  prefix = predfile.split("_data.json")[0]
  result_file = f"{prefix}_edit_dist_by_cefr.txt"
  with open(result_file, "w") as f:
        f.write(json.dumps(metrics))