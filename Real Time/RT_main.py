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

from cc_lin_dist_flow_RT import cc_lin_dist_flow_model


def read_network_data_lfm(file_name):
    baseMVA = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='baseMVA',index_col=None)['baseMVA'].to_list()[0]
    branch_data = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Branch',index_col=0)
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

def load_wind_sc(z, nb_t, nb_scn_distrib, nb_scn_rt):
    wind_sc_z = pd.read_csv('wind_scenarios/wp-scen-zone{}.dat'.format(z), sep = ' ',header=None)
    wind_sc_z = wind_sc_z.iloc[0:nb_t,nb_scn_distrib+1:nb_scn_distrib+1+nb_scn_rt]
    wind_sc_z_array = wind_sc_z.to_numpy()
    return wind_sc_z_array

def wind_real(network_data, nb_scn_rt):
    wf_data = network_data['wf_data']
    zones = (wf_data['Zone'].unique()).tolist() # Retrieve all zones
    wind_scn_all_z = []
    for z in zones:
        wind_scn_z_array = load_wind_sc(z, nb_t, nb_scn_distrib, nb_scn_rt)
        wind_scn_all_z.append(wind_scn_z_array)
    
    wind_scn_per_wf = []
    for scn in range(nb_scn_rt):
        wind_scn = []
        for wf in wf_data.index:
            id_zone = zones.index(wf_data.at[wf,'Zone'])
            wind_rt = wind_scn_all_z[id_zone][0,scn]
            wind_scn.append(wind_rt*wf_data.at[wf,'Pmax'])
        wind_scn_per_wf.append(wind_scn)
    
    return wind_scn_per_wf

def uniform_costs(network_data, cost_A_up, cost_A_down, cost_C, cost_NS):
    costs = {}
    costs_A_up=[]
    costs_A_down=[]
    costs_C=[]
    costs_NS=[]
    for i in range(len(network_data['bus_data'])):
        costs_A_up.append(cost_A_up)
        costs_A_down.append(cost_A_down)
        costs_C.append(cost_C)
        costs_NS.append(cost_NS)
    costs['A+'] = costs_A_up
    costs['A-'] = costs_A_down
    costs['C'] = costs_C
    costs['NS'] = costs_NS
    return costs

if __name__ == '__main__':
    
    #%% MODIFY THE PARAMETERS
    
    nb_t = 1 # Nb of time periods considered
    nb_scn_distrib = 1000 # Nb of realizations used to evaluate the distribution of the error
    nb_scn_rt = 2000 # Nb of realizations used to evaluate the real time costs
    
    check_line_loading = False
    
    network_file = 'network15bus_new.xlsx'
    accepted_files = ['SC','DC_no_zones','DC_conservative_zones','DC_exact_zones']
    test_cases = ['high_liq','med_liq','low_liq','no_liq']
    case_name = '15bus'
    display_results = False
    
    # Power Factor
    pf = 0.95
    
    # Costs
    costs = {}
    cost_A_up = 0
    cost_A_down = 0
    cost_C = 60
    cost_NS = 200
    costs_uniform = True # If false, define costs as lists, with a value for each bus

    #%%
    
    # Unpack network data
    Network_data = read_network_data_lfm(network_file)
    
    if costs_uniform == True:
        costs = uniform_costs(Network_data, cost_A_up, cost_A_down, cost_C, cost_NS)
    wind_scn = wind_real(Network_data, nb_scn_rt)
    
    for a in accepted_files:
        for t in test_cases:
            accepted = pd.read_csv('{}_{}_accepted_{}.csv'.format(case_name,t,a))
            accepted_offers = accepted[accepted['Type']=='Offer']
            print()
            print(a,t)
            print()
    
            i = 0
            for w in wind_scn:
                i+=1
                results = cc_lin_dist_flow_model(Network_data, accepted_offers, w, pf, costs, display_results, check_line_loading, case_name, t, a, i)




