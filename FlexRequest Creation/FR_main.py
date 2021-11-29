"""
Main file for running the FlexRequest model.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import sqrtm

from cc_lin_dist_flow import cc_lin_dist_flow_model
from mc_validation_FR import mc_out_validation

#%% MODIFY THE PARAMETERS

nb_t = 1 # Nb of time periods considered
nb_scn_distrib = 2 # Nb of realizations used to evaluate the distribution of the error
nb_scn_mc = 2 # Nb of realizations used for the Monte Carlo analysis

network_file_name = 'network_15bus.xlsx'
requests_file = 'requests.csv' # Output csv with requests
case_name = '15bus'

# Displaying results for the different steps
display_results_FR = False
display_results_val = False

# Define acceptable violation probabilities for the chance constraints
viol_prob = {}
viol_prob['S'] = 0.05
viol_prob['V'] = 0.05
viol_prob['PF'] = 0.05
beta_factor = 0.5

# Power Factor
pf = 0.95

#%%
def read_network_data_lfm(file_name, nb_t, nb_scn):
    baseMVA = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='baseMVA',index_col=None)['baseMVA'].to_list()[0]
    branch_data = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Branch',index_col=None)
    bus_data =  pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Bus',index_col=None)
    SetPoint = list(bus_data['Setpoint P - t1'])  # Baseline injections at each node (negative for retrieval)
    wf_data = pd.read_excel(open(file_name, 'rb'),engine='openpyxl',sheet_name='Windfarms',index_col=None)
    nb_wf = wf_data.shape[0]
    
    # Wind data
    # Get the point forecasts for wind
    wind_all_pf=[]
    for wf in wf_data.index:
        wind_pf = load_wind_pf(wf_data.at[wf,'Zone'],nb_t) # Load the pointforecast for this windfarm
        wind_all_pf.append(wind_pf[0]*wf_data.at[wf,'Pmax'])
    # Get the realizations
    all_scn = np.zeros((nb_wf*nb_t,nb_scn))
    zones = (wf_data['Zone'].unique()).tolist() # Retrieve all zones
    wind_scn_all_z = []
    for z in zones:
        wind_scn_z_array = load_wind_sc(z, nb_t, nb_scn)
        wind_scn_all_z.append(wind_scn_z_array)
    for scn in range(nb_scn):
        wind_scn = []
        for wf in wf_data.index:
            id_zone = zones.index(wf_data.at[wf,'Zone'])
            wind_rt = wind_scn_all_z[id_zone][0,scn]
            wind_scn.append(- wind_rt*wf_data.at[wf,'Pmax'])
        all_scn[:,scn] = np.add(wind_scn, np.array(wind_all_pf))
    # Get the covariance matrix
    covariance_matrix = np.cov(all_scn)
    # np.linalg.cholesky(covariance_matrix) # To check that the matrix is PSD
    covariance_matrix_squared = sqrtm(covariance_matrix)
    
    Data_network = {}
    Data_network['base_MVA'] = baseMVA
    Data_network['bus_data'] = bus_data
    Data_network['branch_data'] = branch_data
    Data_network['wf_data'] = wf_data
    Data_network['setpoint'] = SetPoint
    Data_network['sq_cov_mtx'] = covariance_matrix_squared
    Data_network['wind_all_pf'] = wind_all_pf

    return Data_network

def load_wind_pf(z, nb_t):
    wind_pf_z = pd.read_csv('Wind Data/wind_pf/wp-meas-zone{}.dat'.format(z), header=None)
    wind_pf_z = wind_pf_z.iloc[0:nb_t,0].tolist()
    return wind_pf_z

def load_wind_sc(z, nb_t, nb_scn):
    wind_sc_z = pd.read_csv('Wind Data/wind_scenarios/wp-scen-zone{}.dat'.format(z), sep = ' ',header=None)
    wind_sc_z = wind_sc_z.iloc[0:nb_t,0:nb_scn]
    wind_sc_z_array = wind_sc_z.to_numpy()
    return wind_sc_z_array

def load_wind_sc_eval(z, nb_t, nb_scn_distrib, nb_scn_rt):
    wind_sc_z = pd.read_csv('Wind Data/wind_scenarios/wp-scen-zone{}.dat'.format(z), sep = ' ',header=None)
    wind_sc_z = wind_sc_z.iloc[0:nb_t,nb_scn_distrib+1:nb_scn_distrib+1+nb_scn_rt]
    wind_sc_z_array = wind_sc_z.to_numpy()
    return wind_sc_z_array

def forecast_error(network_data, nb_scn_mc, nb_t, nb_scn_distrib):
    wind_data = network_data['wf_data']
    wind_pf = network_data['wind_all_pf']
    wind_snc = {}
    forecast_error = {}
    for w in wind_data.index:
        z = wind_data.loc[w,'Zone']
        wind_snc[w] = load_wind_sc_eval(z, nb_t, nb_scn_distrib, nb_scn_mc)
    for s in range(1,nb_scn_mc+1):
        for t in range(1,nb_t+1):
            error = []
            for w in wind_data.index:
                realization = wind_snc[w][t-1,s-1]*wind_data.loc[w,'Pmax']
                error.append(wind_pf[w]-realization)
            forecast_error[s,t] = error
    return forecast_error

def line_node_incidence_matrix(network_data):
    buses_ip = network_data['bus_data']
    lines_ip = network_data['branch_data']
    
    n_buses = buses_ip.shape[0]
    n_lines = lines_ip.shape[0]

    DG = nx.DiGraph() # Create an empty directed graph structure with no nodes and no edges
    G = nx.Graph() # Create an empty graph structure with no nodes and no edges
    for l in lines_ip.index:
        G.add_edge(lines_ip.loc[l, 'From'], lines_ip.loc[l, 'To'])
        DG.add_edge(lines_ip.loc[l, 'From'], lines_ip.loc[l, 'To'])

    root_bus = buses_ip[buses_ip['type'] == 3]['Bus'].to_list()[0] # Return the id of the slack bus
    list_end_buses = [x for x in DG.nodes() if DG.out_degree(x) == 0 and DG.in_degree(x) == 1] # (no edges pointing out of the node, one edge pointing in the node)
    paths_end_buses = {}
    for eb in list_end_buses:
        paths_end_buses[eb] = nx.shortest_path(G, root_bus, eb) # List of nodes in the shortest path from the slack bus to the end bus considered

    pow_inc_mtx = np.zeros((n_lines, n_buses))
    vlt_inc_mtx = np.zeros((n_buses, n_lines))

    for b in G.nodes:
        b_ind = buses_ip[buses_ip['Bus'] == b].index.to_list()[0]
        if buses_ip.loc[b_ind, 'type'] != 3:
            for eb in list_end_buses:
                if b == eb:
                    l = lines_ip[lines_ip['To'] == b].index
                    pow_inc_mtx[l, b_ind] = 1
                else:
                    if b in paths_end_buses[eb]:
                        l = lines_ip[lines_ip['To'] == b].index
                        path = nx.shortest_path(G, b, eb).copy()
                        path_ind = []
                        for p in path:
                            path_ind.extend(buses_ip[buses_ip['Bus'] == p].index.to_list())
                            pow_inc_mtx[l, path_ind] = 1
            path = nx.shortest_path(G, b, root_bus).copy()
            lines_list = []
            for s in path:
                for t in path:
                    lines_list.extend(lines_ip.loc[(lines_ip['From'] == s) & (lines_ip['To'] == t)].index.to_list())
            vlt_inc_mtx[b_ind, lines_list] = 1

    return pow_inc_mtx, vlt_inc_mtx

def uncertainty_incidence_matrix(network_data):
    bus_ip = network_data['bus_data']
    wind_ip = network_data['wf_data']
    all_bus = bus_ip['Bus'].tolist()
    nb_bus = len(all_bus)
    wind_bus = wind_ip['Bus'].tolist()
    nb_wf = len(wind_bus)
    uncertainty_inc_mtx = np.zeros((nb_bus,nb_wf))
    for w in range(nb_wf):
        wb = all_bus.index(wind_bus[w])
        uncertainty_inc_mtx[wb,w] = 1
    return uncertainty_inc_mtx

#%%
if __name__ == '__main__':

    # Unpack network data
    network_file = '{} Data/{}'.format(case_name,network_file_name)
    Network_data = read_network_data_lfm(network_file, nb_t, nb_scn_distrib)

    # Create incidence matrices
    pow_inc_mat, vlt_inc_mat = line_node_incidence_matrix(Network_data)
    uncertainty_inc_mat = uncertainty_incidence_matrix(Network_data)
    
    # Run CC optimization problem to create the FlexRequests
    flex_res = cc_lin_dist_flow_model(case_name, Network_data, pf, pow_inc_mat, vlt_inc_mat, uncertainty_inc_mat, viol_prob, beta_factor, display_results_FR, requests_file)
    
    # Monte Carlo analysis for the performance of the CC model
    # Get the error realizations
    forecast_err = forecast_error(Network_data, nb_scn_mc, nb_t, nb_scn_distrib)
    cc = mc_out_validation(forecast_err, flex_res, Network_data, pow_inc_mat, vlt_inc_mat, uncertainty_inc_mat, nb_scn_mc, nb_t, pf, display_results_val, case_name)