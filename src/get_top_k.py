from datasets import load_dataset
import argparse

"""
Function that gets the n topk keyphrases from a number of generated sequences
"""
def topk(dataset,n=5):
    topk=[]

    # Tant que l'on n'a pas assez de mots-clés ou qu'on n'a pas tout vidé

    for i,kp_list in enumerate(dataset["pred"]): # pour chacune des listes
        if len(kp_list) > 0: # s'il y a au moins un mot-clé dedans
            for kp in kp_list:
                if kp not in topk and kp != '': # s'il n'est pas déjà dans la liste
                    topk.append(kp) # on l'ajoute
                #print(len(topk))
                else:
                    continue
        else: #Si la list est vide on s'en débarasse
            
            continue

    if len(topk) > n: # If we have enough unique keyphrases
        dataset["top_{}".format(n)] = topk[:n]

    elif len(topk) < n: # If we don't have enough keyphrases, we pad with wrong ones
        for j in range(n-len(topk)):
            topk.append("<unk>")
        dataset["top_{}".format(n)] = topk
    else:
        dataset["top_{}".format(n)] = topk
    return dataset

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-data_file")
    parser.add_argument("-output_file")

    args = parser.parse_args()

    d = load_dataset("json",data_files ={"test": args.data_file})

    d = d.map(lambda ex :{"pred": [element.split(";") for element in ex["pred"]]}) #The generated sequences are stored in a list
    d = d.map(lambda ex:{"top_m": ex["pred"][0]})
    d = d.map(topk,fn_kwargs={"n": 5})
    d = d.map(topk,fn_kwargs={"n": 10})


    d = d.remove_columns(["title","abstract","text","pred"])

    d["test"].to_json(args.output_file)