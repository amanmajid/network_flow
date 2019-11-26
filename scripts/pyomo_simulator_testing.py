# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:58:41 2019

@author: amanm
"""
# What to do:
# [1] Just make a simple network flow model for starts and make sure it runs
# [2] Add time index
# [3] Add reservoir behaviour
# [4] Add multi-commodity index

import pandas as pd
import pyomo
import pyomo.opt
import pyomo.environ as pe

node_file           = '../data/nodes.csv'
arc_file            = '../data/arcs.csv'
water_supply_file   = '../data/water_supply.csv'
water_demand_file   = '../data/water_demand.csv'

# ------------------
# Data handling
# ------------------

#--------
# Load data from csv using pandas

# Read in the node_data
node_data = pd.read_csv(node_file)
node_data.set_index(['Node'], inplace=True)
node_data.sort_index(inplace=True)

# Read in the arc_data
arc_data = pd.read_csv(arc_file)
arc_data.set_index(['Start','End'], inplace=True)
arc_data.sort_index(inplace=True)

# Read in the supply_data
supply_data = pd.read_csv(water_supply_file)
supply_data.set_index(['Timestep'], inplace=True)
supply_data.sort_index(inplace=True)

# Read in the demand_data
demand_data = pd.read_csv(water_demand_file)
demand_data.set_index(['Timestep'], inplace=True)
demand_data.sort_index(inplace=True)

#--------
# Prepare data for Pyomo readables

# arc-nodes
node_set         = node_data.index.unique()
arc_set          = arc_data.index.unique()

# supply-demand balance
for row in demand_data.index:
    SupplyState = []
    # if supply exceeds demand
    if supply_data.loc[row].sum(axis=0) > demand_data.loc[row].sum(axis=0):
        demand_data['DummyDemand'].loc[row] = supply_data.loc[row].sum(axis=0) - demand_data.loc[row].sum(axis=0)
        SupplyState.append('SURPLUS')
    #if supply is below demand
    elif supply_data.loc[row].sum(axis=0) < demand_data.loc[row].sum(axis=0):
        supply_data['DummySupply'].loc[row] = demand_data.loc[row].sum(axis=0) - supply_data.loc[row].sum(axis=0)
        SupplyState.append('SHORTAGE')
    else:
        SupplyState.append('SUPPLY = DEMAND')

del arc_file, node_file, water_demand_file, water_supply_file

# --------------------------
# Pyomo
# --------------------------

#--------
# Define model
model           = pe.ConcreteModel()
model.dual      = pe.Suffix( direction=pe.Suffix.IMPORT )

#--------
# Sets
model.nodes = pe.Set( initialize=node_set )
model.arcs  = pe.Set( initialize=arc_set, dimen=2 )

# Nodes in
def NodesIn_init(model, node):
    retval = []
    for (i,j) in model.arcs:
        if j == node:
            retval.append(i)
        return retval
model.NodesIn = pe.Set(model.nodes, initialize=NodesIn_init)

# Nodes out
def NodesOut_init(model, node):
    retval = []
    for (i,j) in model.arcs:
        if i == node:
            retval.append(j)
        return retval
model.NodesOut = pe.Set(model.nodes, initialize=NodesOut_init)

#--------
# Params

# demand data
def demandData_init(model, demand_data):
    init = {}
    for node in model.nodes:
        if(node in demand_data.columns):
            init[node] = demand_data[node].iloc[0]
        else:
            init[node] = 0
    return init
init_demand  = demandData_init(model, demand_data)
model.demand = pe.Param(model.nodes, initialize = init_demand)

# supply data
def supplyData_init(model, supply_data):
    init = {}
    for node in model.nodes:
        if(node in supply_data.columns):
            init[node] = supply_data[node].iloc[0]
        else:
            init[node] = 0
    return init
init_supply  = supplyData_init(model, supply_data)
model.supply = pe.Param(model.nodes, initialize = init_supply)


#--------
# Decision variable
model.flow = pe.Var( model.arcs, within=pe.NonNegativeReals )

# Objective function: minimise cost of flow
def objective_function(model):
    return sum(model.flow[i,j] * arc_data.loc[i,j].Cost for (i,j) in model.arcs)
model.objective_function = pe.Objective(rule=objective_function, sense=pe.minimize)


#--------
# Rules

# [Rule 1]: Enforce lower bound to arc flow
def rule_arcFlow_lower(model, i, j):
    return model.flow[i,j] >= arc_data.loc[i,j].LowerBound
model.arc_lower_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_lower)

# [Rule 2]: Enforce upper bound to arc flow
def rule_arcFlow_upper(model, i, j):
    return model.flow[i,j] <= arc_data.loc[i,j].UpperBound
model.arc_upper_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_upper)

# [Rule 3]: Enforce mass balance
def FlowBalance_rule(model, node):
    return model.supply[node] \
    + sum(model.flow[i, node] for i in model.nodes if (i,node) in model.arcs) \
    - model.demand[node] \
    - sum(model.flow[node, j] for j in model.nodes if (node,j) in model.arcs) \
    == 0
    
model.FlowBalance = pe.Constraint(model.nodes, rule=FlowBalance_rule)

# --------------------------
# Solve
# --------------------------

modelRun = model
solver = pyomo.opt.SolverFactory('gurobi')
results = solver.solve(modelRun)
modelRun.solutions.load_from(results)

print(results.write_yaml())

print('# ----------------------------------------------------------')
print('#   Arc Flows')
print('# ----------------------------------------------------------')
print('')
print('State of water supply: ' + SupplyState[0])
print('')
for startNode,endNode in arc_set:
    flow = modelRun.flow[(startNode,endNode)].value
    print('Flow on arc %s -> %s: %.2f'%(str(startNode), str(endNode), flow))