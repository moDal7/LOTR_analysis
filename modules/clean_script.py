import pandas as pd

def clean_script(script_df, char_df, out_path=None):
    
    # function to remove voiceover from the dataset
    def remove_voiceover(x):
        if "voice" in x:
            X = x.split(" ")
            return X[0]
        else:
            return x
       
    # manual fixes to script 
    script_df["char"]=script_df["char"].apply(lambda x: x.lower())
    script_df["char"]=script_df["char"].apply(lambda x: x.strip())
    script_df["char"]=script_df["char"].apply(lambda x: x.strip("("))
    script_df["char"]=script_df["char"].apply(remove_voiceover)

    # manual fixes for characters' names
    script_df["char"]=script_df["char"].replace("gan dalf", "gandalf")
    script_df["char"]=script_df["char"].replace("eowyn/merry", "eowyn")
    script_df["char"]=script_df["char"].replace("eye of sauron", "sauron")
    script_df["char"]=script_df["char"].replace("galadril", "galadriel")
    script_df["char"]=script_df["char"].replace("argorn", "aragorn")
    script_df["char"]=script_df["char"].replace("strider", "aragorn")
    script_df["char"]=script_df["char"].replace("voice", "gollum")
    script_df["char"]=script_df["char"].replace("ring", "sauron")
    script_df["char"]=script_df["char"].replace("white wizard", "saruman")
    script_df["char"]=script_df["char"].replace("grishnak", "grishnakh")
    script_df["char"]=script_df["char"].replace("merry and pippin", "merry")
    script_df["char"]=script_df["char"].replace("gatekeepr", "gatekeeper")

    # manual fixes for other characters functions to remove errors/ wrong entries from the dataset
    # different names
    char_df["name"]=char_df["name"].replace("aragorn ii elessar", "aragorn")
    char_df["name"]=char_df["name"].replace("bilbo baggins", "bilbo")
    char_df["name"]=char_df["name"].replace("déagol", "deagol")
    char_df["name"]=char_df["name"].replace("denethor ii", "denethor")
    char_df["name"]=char_df["name"].replace("omer", "eomer")
    char_df["name"]=char_df["name"].replace("othain", "eothain")
    char_df["name"]=char_df["name"].replace("owyn", "eowyn")
    char_df["name"]=char_df["name"].replace("frodo baggins", "frodo")
    char_df["name"]=char_df["name"].replace("meriadoc brandybuck", "merry")
    char_df["name"]=char_df["name"].replace("samwise gamgee", "sam")
    char_df["name"]=char_df["name"].replace("peregrin took", "pippin")
    char_df["name"]=char_df["name"].replace("rosie cotton", "rosie")
    char_df["name"]=char_df["name"].replace("théoden", "theoden")
    char_df["name"]=char_df["name"].replace("barliman butterbur", "barliman")
    char_df["name"]=char_df["name"].replace("gríma wormtongue", "grima")
    char_df["name"]=char_df["name"].replace("witch-king of angmar", "witch king")
    char_df["name"]=char_df["name"].replace("the king of the dead", "king of the dead")
    char_df["name"]=char_df["name"].replace("iorlas", "irolas")
    char_df["name"]=char_df["name"].replace("haldir (lorien)", "haldir")
    char_df["name"]=char_df["name"].replace("háma", "hama")
    char_df["name"]=char_df["name"].replace("uglúk", "ugluk")
    char_df["name"]=char_df["name"].replace("grishnákh", "grishnakh")
    char_df["name"]=char_df["name"].replace("old noakes", "noakes")

    # duplicate races
    char_df["race"]=char_df["race"].replace("Hobbit", "Hobbits")

    # manual smeagol fix, duplicate of gollum with changed name
    char_df.loc[911]=["TA 2430", "March 25 ,3019", "Male", "NaN", "NaN", "smeagol", "Hobbits", "NaN", "NaN"]
    char_df["name"]=char_df["name"].apply(lambda x: x.lower().strip())

    # merge script and character dataframes
    lotr_df = script_df.merge(char_df, left_on="char", right_on="name", how="left", indicator=True)

    # simplify races
    lotr_df.loc[lotr_df["char"]=="orc", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="orcs", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="captain", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="general", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="sharku", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="army", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="uruk hai", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="uruk-hai", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="gothmog", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="hobbit", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="children hobbits", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="proudfoot hobbit", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="mrs bracegirdle", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="gaffer", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="sandyman", "race"]="Hobbits"

    # assign races to characters that do not have one
    lotr_df.loc[lotr_df["char"]=="soldiers on gate", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="soldiers in minas tirith", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="soldiers", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="soldier", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="soldier 1", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="soldier 2", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="boson", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="mercenary", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="rohirrim", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="madril", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="woman", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="old man", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="crowd", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="wildman", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="people", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="lady", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="freda", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="rohan stableman", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="men", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="gatekeeper", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="rohan horseman", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="general shout", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="man", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="aragorn", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="deagol", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="figwit", "race"]="Elves"
    lotr_df.loc[lotr_df["char"]=="merry", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="pippin", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="sam", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="bilbo", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="haldir", "race"]="Elves"
    lotr_df.loc[lotr_df["char"]=="noakes", "race"]="Ents"
    lotr_df.loc[lotr_df["char"]=="rosie", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="frodo", "race"]="Hobbits"
    lotr_df.loc[lotr_df["char"]=="denethor", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="theoden", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="grima", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="witch king", "race"]="Wraith/Undead"
    lotr_df.loc[lotr_df["char"]=="king of the dead", "race"]="Wraith/Undead"
    lotr_df.loc[lotr_df["char"]=="hama", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="irolas", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="barliman", "race"]="Men"
    lotr_df.loc[lotr_df["char"]=="ugluk", "race"]="Uruk-hai/Orcs"
    lotr_df.loc[lotr_df["char"]=="grishnakh", "race"]="Uruk-hai/Orcs"

    lotr_df["race"]=lotr_df["race"].replace("Uruk-Hai,Orc", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Uruk-hai,Orc", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Uruk-Hai", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Uruk-hai", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Orc", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Orcs", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Black Uruk", "Uruk-hai/Orcs")
    lotr_df["race"]=lotr_df["race"].replace("Half-elven", "Elves")
    lotr_df["race"]=lotr_df["race"].replace("Half-elven,Men", "Elves")
    lotr_df["race"]=lotr_df["race"].replace("Ents,Onodrim", "Ents")
    lotr_df["race"]=lotr_df["race"].replace("Men,Undead", "Wraith/Undead")
    lotr_df["race"]=lotr_df["race"].replace("Men,Wraith", "Wraith/Undead")

    # fix missing fields
    lotr_df.loc[lotr_df["gender"].isnull(), "gender"]="Other"
    lotr_df.loc[lotr_df["dialog"].isnull(), "dialog"]=" "

    # fix multiple spaces and starting spaces
    lotr_df["dialog"]=lotr_df["dialog"].apply(lambda x: " ".join(x.split()))

    #remove rows if dialog is empty
    lotr_df=lotr_df[lotr_df["dialog"]!=" "]
    lotr_df=lotr_df[lotr_df["dialog"]!=""]

    lotr_df=lotr_df.drop(columns=["Unnamed: 0", "name", "_merge", "spouse", "realm", "hair", "height"])
    lotr_df=lotr_df[["char", "dialog", "movie", "birth", "death", "gender", "race"]]

    lotr_df["char"]=lotr_df["char"].apply(lambda x: x.upper())

    if out_path is None:
        lotr_df.to_csv("../data/lotr_dataset.csv", index=True)
    else:
        lotr_df.to_csv(out_path, index=True)

    print("script.csv cleaned and saved to data/lotr_dataset.csv")
    return lotr_df, script_df, char_df

if __name__=="__main__":
    print("clean_script.py is being run directly")
    script_df=pd.read_csv("../data/lotr_scripts.csv")
    char_df=pd.read_csv("../data/lotr_characters.csv")
    clean_script(script_df, char_df);
    
