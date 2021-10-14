"""
Main file for running the FlexRequest model.
It includes the following utils functions:
1) read_network_data_lfm - specialized for the 15-bus IEEE radial distribution system
2) line_node_incidence_matrix - to create incidence matrices for the connection between lines & bus injections
and bus & line flows
3) cc_margins - create uncertainty margins assumming a Gaussian pdf of the uncertainty with zero mean

09.06.2021 ID
"""

import pandas as pd

from Auction_Deterministic import deterministic_mc


def read_network_data_lfm(file_name):
    baseMVA = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='baseMVA',index_col=None)['baseMVA'].to_list()[0]
    branch_data = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Branch',index_col=None)
    bus_data =  pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Bus',index_col=None)
    SetPoint = list(bus_data['Setpoint P - t1'])  # Baseline injections at each node (negative for retrieval)
    wf_data = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Windfarms',index_col=None)
    
    Data_network = {}
    Data_network['base_MVA'] = baseMVA
    Data_network['bus_data'] = bus_data
    Data_network['branch_data'] = branch_data
    Data_network['wf_data'] = wf_data
    Data_network['setpoint'] = SetPoint

    return Data_network

if __name__ == '__main__':
    
    #%% MODIFY THE PARAMETERS
    
    nb_t = 1 # Nb of time periods considered
    
    # network_file = 'network_bnNETZE_V7.xlsx'
    # offers_file = 'offers_bnNETZE.csv'
    # requests_file = 'requests_bnNETZE.csv' # Output csv with requests, used as input for market clearing
    # zones_file = 'dso_zones_bnNETZE.csv'
    # accepted_file = 'Accepted_Deterministic_bnNETZE.csv' # Output csv with accepted offers
    
    case_name = '15bus'
    network_file = 'network15bus_new.xlsx'
    offers_files = ['high_liq','med_liq','low_liq','no_liq']
    requests_file = 'requests'
    zones_files = ['no_zones', 'conservative_zones', 'exact_zones']
    
    # Displaying results for the different steps
    display_results_MC = False
    
    # Define acceptable violation probabilities for the chance constraints
    viol_prob = {}
    viol_prob['S'] = 0.05
    viol_prob['V'] = 0.05
    viol_prob['PF'] = 0.05
    beta_factor = 0.5
    
    # Power Factor
    pf = 0.95
    
    # Prices for FlexRequests
    c_flex_up = 70
    c_flex_down = 40

#%%
    # Unpack network data
    Network_data = read_network_data_lfm(network_file)
    
    # Run the market clearing
    res = {}
    for o in offers_files:
        for z in zones_files:
            res[o,z] = deterministic_mc(case_name, network_file, o, requests_file, c_flex_up, c_flex_down, z, display_results_MC)