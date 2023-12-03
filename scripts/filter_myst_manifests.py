import re

in_file = "/workspace/datasets/myst_w2v2_asr/train_manifest.json"
out_file = "/workspace/datasets/myst_w2v2_asr/train_manifest2.json"
with open(in_file, "r") as f1:
        with open(out_file, "w") as f2:
                for line in f1:
                        # transcript = line.split("text\": \"")[1].split("\"")[0]
                        if "\\" in line:
                                # replace \\u00e2\\u20ac\\u2122 with ' char
                                # replace \u00e2\u20ac\u02dc with ' char
                                # remove \\u00e2\\u00a0
                                # remove \\u00e2\\u20ac\\u201c
                                # remove \u00e2\u20ac\u00a6
                                line = line.replace("\\u00e2\\u20ac\\u2122", "'")
                                line = line.replace(r"\u00e2\u20ac\u02dc", "'")
                                line = line.replace("\\u00e2\\u00a0", " ")
                                line = line.replace("\\u00e2\\u20ac\\u201c", " ")
                                line = line.replace(r"\u00e2\u20ac\u00a6", " ")
                        line = re.sub(' +', ' ', line)
                        f2.write(line)

# MANUAL POSTPROCESSING STEPS:
# 1: in line 9818 of new /workspace/datasets/myst_w2v2_asr/train_manifest.json remove space before ['ve].
# 2: in line 10545 of new /workspace/datasets/myst_w2v2_asr/train_manifest.json remove trailing whitespace in the text value field.