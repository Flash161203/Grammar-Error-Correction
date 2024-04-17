# Command to execute

1. Paste csv files containing src, hyp and ref sentences each into the folder
2. Execute `./compute_gleu -s src_all.csv -r ref_all.csv  -o output_all.csv --debug` (or remove --debug flag to hide sentence-level scores)
3. Ignore the warning, since we only have one reference per sentence (see [this](https://github.com/keisks/jfleg/issues/2))

Adapted from [gec-ranking github](https://github.com/cnap/gec-ranking/blob/master/scripts/gleu.py?tab=readme-ov-file)
