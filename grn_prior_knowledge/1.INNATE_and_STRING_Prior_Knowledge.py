
##############################################################
#
# IMPORT MODULES
#
##############################################################

import re
import requests
import mygene
from urllib.request import urlopen
import stringdb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")



##############################################################
#
# IMPORT DATA AND SET IMPORTANTE VARIABLES
#
##############################################################

Relief_variables = pd.read_csv("C:/Users/Inigo/Documents/relief_variables.csv", sep=',') # Loads the selection of genes
IDs = Relief_variables.drop('Unnamed: 0', axis=1).iloc[1,] # Keeps only the gene IDs

Prior_knowledge = pd.DataFrame(0, index=IDs, columns=IDs) # Generate the dataframe filled with 0 to save the prior knowledge gene interactions

mg = mygene.MyGeneInfo() # Loads mygene function to transform ensembl ids to gene symbols
server = "https://rest.ensembl.org" # Loads ensembl connection
string_server = "http://psicquic.curated.innatedb.com/webservices/current/search/query/" # Loads psicquic innate curated connection



##############################################################
#
# INNATE DEFINITION
#
##############################################################

def process_external_data(gene):
    try:
        url_str = f"{string_server}{gene}" # Set the gene search in database
        with urlopen(url_str) as file_handle: # Perform the search in database
            content = file_handle.read().decode('utf-8') # Save the records of the search
        lines = content.splitlines() # Split the records found for the gene
        interactions = [] # Generate a empty dataframe to save the interactions found
        for line in lines: # Iterate over all interactions found for the gene
            idx1 = [m.end() for m in re.finditer('ensembl:', line)] # Substract the ensembl ids records in the interaction information
            idx2 = [m.start() for m in re.finditer('\t', line)] # Substract the separations of information inside an interaction record

            idx2_correct = [min(filter(lambda x: x > i, idx2), default=None) for i in idx1] # Substract the separation just after the ensembl ids records
            interactions.extend([line[idx1[i]:idx2_correct[i]] for i in range(len(idx1))]) # Substract the ensembl ids in each interaction record for the gene
        interactions = [item for item in interactions if gene not in item] # Delete the ensembl id of the gene that has been search
        interactions = [d for d in IDs if d in interactions] # Keep the genes ids that are in the list of genes set for the prior knowledge
        for connection_I in interactions: # Iterate over all genes found
            Prior_knowledge.loc[gene, connection_I] = Prior_knowledge.loc[gene, connection_I] + 1.0 # Sum 1 in the prior knowledge table position for that interaction
    except IOError: # If there is not any record for that gene
        print('Cannot open URL ' + url_str) # Make sure you notice of the issue to further investigation
        content = ''

##############################################################
#
# STRING DEFINITION
#
##############################################################

def process_string_data(gene):
    try:
        string_ids = stringdb.get_interaction_partners([gene]) # Seach for gene interactions in STRING
        ensembl_ids = [] # Generate an empty dataframe to save interaction ensembl ids
        gene_names = [] # Generate an empty dataframe to save interaction gene symbols
        for gene_name in string_ids['preferredName_B']: # Iterate over the gene symbols of interaction genes
            try:
                ext = f"/xrefs/symbol/homo_sapiens/{gene_name}?" # Set the seacrh for gene symbol
                r = requests.get(server + ext, headers={"Content-Type": "application/json"}) # Make the search for gene symbol to ensembl id tranformation
                decoded = r.json() # Save results
                gene_names.append(gene_name) # Save the gene symbols
                ensembl_ids.append(decoded[0]['id']) # Save the ensembl ids
            except IndexError: # If there is no record for the gene symbol to ensembl id tranformation
                ensembl_ids.append('No found') # Make sure the code do not stop
            except requests.exceptions.SSLError: # If there is a connection error to the database
                ensembl_ids.append('No found') # Make sure the code do not stop
        ids = pd.concat([string_ids, pd.Series(ensembl_ids)], axis=1) # Save ensembl ids with score table
        STRING_o = ids.loc[ids[0].isin(set([d for d in IDs if d in ensembl_ids]))] # Keep the genes ids that are in the list of genes set for the prior knowledge
        for connection_II in STRING_o[0]: # Iterate over the gene interactions of the genes
            Prior_knowledge.loc[gene, connection_II] = Prior_knowledge.loc[gene, connection_II] + np.mean(STRING_o.loc[STRING_o[0] == connection_II, 'score'])
    except ValueError: # If there is no record for the enembl id format gene in STRING
        try:
            gene_symbol = mg.querymany(gene, scopes='ensembl.gene', fields='symbol', species='human')[0]['symbol'] # Transform the gene to gene symbol
            string_ids = stringdb.get_interaction_partners([gene_symbol]) # Try the search again with the gene symbol
            ensembl_ids = []
            gene_names = []
            for gene_name in string_ids['preferredName_B']:
                try:
                    ext = f"/xrefs/symbol/homo_sapiens/{gene_name}?"
                    r = requests.get(server + ext, headers={"Content-Type": "application/json"})
                    decoded = r.json()
                    gene_names.append(gene_name)
                    ensembl_ids.append(decoded[0]['id'])
                except IndexError:
                    ensembl_ids.append('No found')
                except requests.exceptions.SSLError:
                    ensembl_ids.append('No found')
            ids = pd.concat([string_ids, pd.Series(ensembl_ids)], axis=1)
            STRING_o = ids.loc[ids[0].isin(set([d for d in IDs if d in ensembl_ids]))]
            for connection_II in STRING_o[0]:
                Prior_knowledge.loc[gene, connection_II] = Prior_knowledge.loc[gene, connection_II] + np.mean(STRING_o.loc[STRING_o[0] == connection_II, 'score'])
        except ValueError:
            print('Cannot find gene in STRING ' + gene_symbol)

##############################################################
#
# PARALLEL PROCESSING
#
##############################################################

from concurrent.futures import ThreadPoolExecutor # Import module for parallel processing

with ThreadPoolExecutor() as executor: # Executes parallel processing
    executor.map(process_external_data, IDs) # One way innate definition applied to all genes
    executor.map(process_string_data, IDs) # Second way STRING definition applied to all genes

##############################################################
#
# SAVE PRIOR KNOWLEDGE TABLE
#
##############################################################

#sum(Prior_knowledge.sum(axis=1)) # Check for interaction detections
#Prior_knowledge.sum(axis=1) # Check for interactions distribution
Prior_knowledge.to_csv("C:/Users/Inigo/Documents/PK_final.csv") # Save table with indexes and colnames in csv

##############################################################
#
# GENERATE PRIOR KNOWLEDGE TABLE
#
##############################################################

file_paths = ["C:/Users/Inigo/Documents/PK_final_{}.csv".format(i) for i in range(2000, 20001, 2000)] # Directory to all tables
PK_tables = [pd.read_csv(file_path, sep=',') for file_path in file_paths] # Load all tables
PK_table = pd.concat(PK_tables) # Merge all tables
Prior_knowledge_PRECISEADS = PK_table.set_index(PK_table.columns[0]) # Set IDsas indexes

sum(Prior_knowledge_PRECISEADS.sum(axis=1)) # Check for interaction detections
Prior_knowledge_PRECISEADS.sum(axis=1) # Check for interactions distribution
sum(Prior_knowledge_PRECISEADS.sum(axis=1)>0) # Check for efficiency of prior knowledge search

#Prior_knowledge_PRECISEADS.to_csv("C:/Users/Inigo/Documents/Prior_Knowledge_PRECISEADS.csv") # Save final prior knowledge table