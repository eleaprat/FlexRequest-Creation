# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:02:23 2021

@author: emapr
"""

import gurobipy as gb
import pandas as pd

def deterministic_mc(case_name, network_file, offers_file, requests_file, c_flex_up, c_flex_down, zones_file, display_results):
    
    if display_results == True:
        print('----------------------Deterministic Market Clearing----------------------')

    # Definition of the parameters and sets
    
    nodes_df = pd.read_excel(open(network_file, 'rb'),engine='openpyxl',sheet_name='Bus',index_col=0)
    nodes = list(nodes_df.index)         # index for nodes
    # Retrieve slack bus
    ref = list(nodes_df[nodes_df['type'] == 3].index)[0] # Return the id of the slack bus
    
    offers_df = pd.read_csv(open('{}.csv'.format(offers_file), 'rb'))
    offers = list(offers_df.index)         # index for offers
    
    requests_df = pd.read_csv(open('{}.csv'.format(requests_file), 'rb'))
    for r in requests_df.index:
        if requests_df.loc[r,'Direction']=='Up':
            requests_df.loc[r,'Price'] = c_flex_up
        elif requests_df.loc[r,'Direction']=='Down':
            requests_df.loc[r,'Price'] = c_flex_down
    requests = list(requests_df.index)         # index for offers
    
    zones_df = pd.read_csv(open('{}.csv'.format(zones_file), 'rb'),index_col=0)
    zones = list(set(list(zones_df['Zone'])))    # index for zones
    
    # Model alias
    model = gb.Model('Deterministic_Clearing')
    
    # Display model output: 0=No, 1=Yes
    if display_results == True:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0
    
    #%%  Definition of the variables
    
    # FlexRequests matching
    P_req = {}
    for r in requests:
        P_req[r] = model.addVar(lb=0, ub=requests_df.loc[r,'Quantity'], name='P_req_({})'.format(r))
            
    # FlexOffers matching
    P_off = {}
    for o in offers:
        P_off[o] = model.addVar(lb=0, ub=offers_df.loc[o,'Quantity'], name='P_off_({})'.format(o))
      
    # Update of the model with the variables
    model.update()
    
    #%% Objective function
    
    # Maximize social welfare
    obj = gb.LinExpr()
    # obj.add(gb.quicksum(offers_df.loc[o,'Price']*P_off[o] for o in offers))
    obj.add(gb.quicksum(-requests_df.loc[r,'Price']*P_req[r] for r in requests))
    model.setObjective(obj,gb.GRB.MINIMIZE)
    
    #%% Constraints
    
    
    ED_up = {}
    ED_down = {}
    
    # Power balance up
    for z in zones:
        offers_z = gb.LinExpr()
        requests_z = gb.LinExpr()
        for i in nodes:
            if zones_df.at[i,"Zone"] == z:
                for o in offers:
                    if offers_df.loc[o,'Bus'] == i and offers_df.loc[o,'Direction'] == 'Up':
                        offers_z.add(P_off[o])
                for r in requests:
                    if requests_df.loc[r,'Bus'] == i and requests_df.loc[r,'Direction'] == 'Up':
                        requests_z.add(P_req[r])
        ED_up[z] = model.addConstr(offers_z,gb.GRB.EQUAL,requests_z,name='ED_Up_({})'.format(z))
        
    # Power balance down
    for z in zones:
        offers_z = gb.LinExpr()
        requests_z = gb.LinExpr()
        for i in nodes:
            if zones_df.at[i,"Zone"] == z:
                for o in offers:
                    if offers_df.loc[o,'Bus'] == i and offers_df.loc[o,'Direction'] == 'Down':
                        offers_z.add(P_off[o])
                for r in requests:
                    if requests_df.loc[r,'Bus'] == i and requests_df.loc[r,'Direction'] == 'Down':
                        requests_z.add(P_req[r])
        ED_down[z] = model.addConstr(offers_z,gb.GRB.EQUAL,requests_z,name='ED_Down_({})'.format(z))
        
    model.update()    # update of the model with the constraints and objective function
    
    #%%  Optimization and Results
    
    model.optimize()
    
    #Results display
    
    if display_results == True:
        model.write('Auction_Deterministic_Model.lp') # Write the model
        model.write('Auction_Deterministic_Solutions.sol') # Write the results
    
    points={}
    
    feas=model.Status
    
    if feas == 2: # If the problem is feasible
        Accepted_df = pd.DataFrame(columns=['Type','Bus','Direction','Quantity','Price','Time_target','Market_Price'])
        for v in model.getVars():
            points[v.varName] =  v.x # Value of the primal variables
        for c in model.getConstrs():
            points[c.ConstrName] = c.pi # Value of the dual variables
        for o in offers:
            Poff = points['P_off_({})'.format(o)]
            if Poff != 0:
                market_price = points['ED_{}_({})'.format(offers_df.loc[o,'Direction'],zones_df.at[offers_df.loc[o,'Bus'],'Zone'])]
                new_row = {"Type":'Offer', "Bus":offers_df.loc[o,'Bus'], "Direction":offers_df.loc[o,'Direction'], "Quantity":Poff, "Price":offers_df.loc[o,'Price'], "Time_target":offers_df.loc[o,'Time period'], "Market_Price":market_price}
                Accepted_df = Accepted_df.append(pd.Series(new_row, name=o), ignore_index=False)
                Accepted_df.index.name = "ID"
        for r in requests:
            Preq = points['P_req_({})'.format(r)]
            if Preq != 0:
                market_price = points['ED_{}_({})'.format(requests_df.loc[r,'Direction'],zones_df.at[requests_df.loc[r,'Bus'],'Zone'])]
                new_row = {"Type":'Request', "Bus":requests_df.loc[r,'Bus'], "Direction":requests_df.loc[r,'Direction'], "Quantity":Preq, "Price":requests_df.loc[r,'Price'], "Time_target":requests_df.loc[r,'Time period'], "Market_Price":market_price}
                Accepted_df = Accepted_df.append(pd.Series(new_row, name=r), ignore_index=False)
                Accepted_df.index.name = "ID"
            Accepted_df.to_csv('{}_{}_accepted_DC_{}.csv'.format(case_name, offers_file, zones_file))
    else: # If the problem is infeasible, retrieve the optimization status only
        points['feas'] = feas
    
    # Export costs results 
    file_name = '{}_results_procurement.csv'.format(case_name)
    costs_df = pd.read_csv(file_name, index_col=None)
    SW = 0
    dso_costs = 0
    for o in offers:
        i = offers_df.loc[o,'Bus']
        if offers_df.loc[o,'Direction']=='Up':
            cost_req = c_flex_up
        elif offers_df.loc[o,'Direction']=='Down':
            cost_req = c_flex_down
        SW += points['P_off_({})'.format(o)]* (cost_req - offers_df.loc[o,'Price'])
        dso_costs += points['P_off_({})'.format(o)]* cost_req
    if offers_file == 'high_liq':
        liq = 'High'
    elif offers_file == 'med_liq':
        liq = 'Medium'
    elif offers_file == 'low_liq':
        liq = 'Low'
    elif offers_file == 'no_liq':
        liq = 'No Offers'
    if zones_file == 'no_zones':
        zones_name = 'No Zones'
    elif zones_file == 'conservative_zones':
        zones_name = 'Conservative Zones'
    elif zones_file == 'exact_zones':
        zones_name = 'Exact Zones'
    new_row={'Social Welfare': SW, 'DSO Costs': dso_costs, 'Liquidity':liq, 'Test Case':zones_name}
    costs_df = costs_df.append(pd.Series(new_row), ignore_index=True)
    costs_df.to_csv(file_name, index=False)
    
    # Export data per bus
    file_name = 'results_bids_{}_{}.csv'.format(case_name, offers_file)
    bids_df = pd.read_csv(file_name, index_col=0)
    for i in nodes:
        if i != ref:
            Poff_up_bid = sum(offers_df.loc[o,'Quantity'] for o in offers if offers_df.loc[o,'Bus']==i and offers_df.loc[o,'Direction']=='Up')
            Poff_down_bid = sum(offers_df.loc[o,'Quantity'] for o in offers if offers_df.loc[o,'Bus']==i and offers_df.loc[o,'Direction']=='Down')
            Preq_up_bid = sum(requests_df.loc[r,'Quantity'] for r in requests if requests_df.loc[r,'Bus']==i and requests_df.loc[r,'Direction']=='Up')
            Preq_down_bid = sum(requests_df.loc[r,'Quantity'] for r in requests if requests_df.loc[r,'Bus']==i and requests_df.loc[r,'Direction']=='Down')
            Poff_up = sum(points['P_off_({})'.format(o)] for o in offers if offers_df.loc[o,'Bus']==i and offers_df.loc[o,'Direction']=='Up')
            Preq_up = sum(points['P_req_({})'.format(r)] for r in requests if requests_df.loc[r,'Bus']==i and requests_df.loc[r,'Direction']=='Up')
            Poff_down = sum(points['P_off_({})'.format(o)] for o in offers if offers_df.loc[o,'Bus']==i and offers_df.loc[o,'Direction']=='Down')
            Preq_down = sum(points['P_req_({})'.format(r)] for r in requests if requests_df.loc[r,'Bus']==i and requests_df.loc[r,'Direction']=='Down')
            if Poff_down < 0.00001:
                Poff_down == 0
            if Preq_down < 0.00001:
                Preq_down == 0
            if Poff_up < 0.00001:
                Poff_up == 0
            if Preq_up < 0.00001:
                Preq_up == 0
            bids_df.at[i, 'Accepted Offer Up - {}'.format(zones_name)] = round(Poff_up,5)
            bids_df.at[i, 'Accepted Offer Down - {}'.format(zones_name)] = round(Poff_down,5)
            bids_df.at[i, 'Accepted Request Up - {}'.format(zones_name)] = round(Preq_up,5)
            bids_df.at[i, 'Accepted Request Down - {}'.format(zones_name)] = round(Preq_down,5)
            bids_df.at[i, 'Request Up'] = round(Preq_up_bid,5)
            bids_df.at[i, 'Request Down'] = round(Preq_down_bid,5)
            bids_df.at[i, 'Offer Up'] = round(Poff_up_bid,5)
            bids_df.at[i, 'Offer Down'] = round(Poff_down_bid,5)
    bids_df.to_csv(file_name)
        
    return points