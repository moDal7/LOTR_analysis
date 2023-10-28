import pandas as pd

def lines_to_txt(csv_path, char_name=None, movie_name=None, txt_path=None, no_char=False):
    """
    Takes a csv file with a column of lines and writes them to a txt file.
    """

    # choose the right output path based on input
    if no_char:
        txt_path="data/lotr_lines_nochar.txt"
    elif txt_path is None:
        if char_name is not None:
            if movie_name is not None:
                txt_path="data/"+char_name.lower()+"_"+movie_name.lower()+"_lines.txt"
            else:
                txt_path="data/"+char_name.lower()+"_lines.txt"
        else:
            if movie_name is not None:
                txt_path="data/"+movie_name.lower()+"_lines.txt"
            else:
                txt_path="data/lotr_lines.txt"
   
    df=pd.read_csv(csv_path)
    df=df[char_name==df["char"]] if char_name is not None else df
    df=df[movie_name==df["movie"]] if movie_name is not None else df
    lines=df["dialog"].tolist()
    characters=df["char"].tolist()
    with open(txt_path, "w") as f:
        for i, line in enumerate(lines):
            if char_name is None and not no_char:
                f.write(characters[i]+": "+"\n")
            f.write(line+"\n")

    print("Lines written to "+txt_path)