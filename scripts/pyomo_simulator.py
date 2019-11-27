"""

    pyomo_simulator.py
    
    Description:
        A network simulation model implemented in Pyomo.
    
    25/11/2019
    Author: Aman Majid

"""

# --------------------------
# Import modules
# --------------------------

import pandas as pd
import pyomo
import pyomo.opt
import pyomo.environ as pe

# --------------------------
# Define model class
# --------------------------

class waterSimulator():

    def __init__(self, node_file='../data/nodes.csv', arc_file='../data/arcs.csv', supply_data_file='../data/water_supply.csv', demand_data_file='../data/water_demand.csv'):
        
        '''
         init
        '''
        
        # Read in the node_data
        self.node_data = pd.read_csv(node_file)
        self.node_data.set_index(['Node'], inplace=True)
        self.node_data.sort_index(inplace=True)

        # Read in the arc_data
        self.arc_data = pd.read_csv(arc_file)
        self.arc_data.set_index(['Start','End'], inplace=True)
        self.arc_data.sort_index(inplace=True)

        # Read in the supply data
        self.supply_data = pd.read_csv(supply_data_file)
        self.supply_data.set_index(['Timestep'], inplace=True)
        self.supply_data.sort_index(inplace=True)
        
        # Read in demand data
        self.demand_data = pd.read_csv(demand_data_file)
        self.demand_data.set_index(['Timestep'], inplace=True)
        self.demand_data.sort_index(inplace=True)
        
        
        #--------
        # Prepare data for Pyomo readables
        
        self.node_set = self.node_data.index.unique()
        self.arc_set = self.arc_data.index.unique()
        
        #--------
        # Balance supply and demand
        #   excess/shortages captured by dummy nodes
        
        for row in self.demand_data.index:
            self.SupplyState = []
            # if supply exceeds demand
            if self.supply_data.loc[row].sum(axis=0) > self.demand_data.loc[row].sum(axis=0):
                self.demand_data['DummyDemand'].loc[row] = self.supply_data.loc[row].sum(axis=0) - self.demand_data.loc[row].sum(axis=0)
                self.SupplyState.append('SURPLUS')
            #if supply is below demand
            elif self.supply_data.loc[row].sum(axis=0) < self.demand_data.loc[row].sum(axis=0):
                self.supply_data['DummySupply'].loc[row] = self.demand_data.loc[row].sum(axis=0) - self.supply_data.loc[row].sum(axis=0)
                self.SupplyState.append('SHORTAGE')
            else:
                self.SupplyState.append('SUPPLY = DEMAND')
   
     
    def pyomo_model_create(self):
        
        '''
         function to create Pyomo (pe) model
             - defined as a ConcreteModel()
        '''

        model           = pe.ConcreteModel()
        model.dual      = pe.Suffix( direction=pe.Suffix.IMPORT )

        # ------
        # Sets
        # ------
        
        # nodes
        model.nodes = pe.Set( initialize=self.node_set )
        
        # arcs
        model.arcs  = pe.Set( initialize=self.arc_set, dimen=2 )
        
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

        # ------
        # Params
        # ------
        
        # demand data
        def demandData_init(model, demand_data):
            init = {}
            for node in model.nodes:
                if(node in demand_data.columns):
                    init[node] = demand_data[node].iloc[0]
                else:
                    init[node] = 0
            return init
        init_demand  = demandData_init(model, self.demand_data)
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
        init_supply  = supplyData_init(model, self.supply_data)
        model.supply = pe.Param(model.nodes, initialize = init_supply)
        
        
        # ------
        # Decision variable
        # ------
        
        # flow across each arc
        model.flow = pe.Var( model.arcs, within=pe.NonNegativeReals )


        # ------
        # Objective function
        # ------

        # Minimise total cost of flow
        def objective_function(model):
            return sum(self.arc_data.loc[i,j].Cost * model.flow[i,j] for (i,j) in model.arcs)
        model.totalCost = pe.Objective(rule=objective_function, sense=pe.minimize)


        # ------
        # Rules
        # ------

        # [Rule 1]: Enforce upper bound to arc flow
        def rule_arcFlow_upper(model, i, j):
            return model.flow[i,j] <= self.arc_data.loc[i,j].UpperBound
        model.arc_upper_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_upper)

        # [Rule 2]: Enforce lower bound to arc flow
        def rule_arcFlow_lower(model, i, j):
            return model.flow[i,j] >= self.arc_data.loc[i,j].LowerBound
        model.arc_lower_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_lower)

        # [Rule 3]: Enforce mass balance at each node
        def FlowBalance_rule(model, node):
            return model.supply[node] \
            + sum(model.flow[i, node] for i in model.nodes if (i,node) in model.arcs) \
            - model.demand[node] \
            - sum(model.flow[node, j] for j in model.nodes if (node,j) in model.arcs) \
            == 0       
        model.FlowBalance = pe.Constraint(model.nodes, rule=FlowBalance_rule)
        
        self.model = model



    def pyomo_model_solve(self, solver='gurobi', tee=False):
        
        '''
         function to solve Pyomo (pe) model
             - solved using gurobi as default
        '''
        
        solver = pyomo.opt.SolverFactory(solver)

        print('-----Solving Pyomo Model-----')
        self.results = solver.solve(self.model)

        # Check that we actually computed an optimal solution, load results
        if (self.results.solver.status != pyomo.opt.SolverStatus.ok):
            print('Check solver not ok?')
        if (self.results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            print('Check solver optimality?')
        
        self.model.solutions.load_from(self.results)


    def pyomo_model_print_solutions(self):
        print(self.results.write_yaml())
        print('# ----------------------------------------------------------')
        print('#   Arc Flows')
        print('# ----------------------------------------------------------')
        print('')
        print('State of water supply: ' + self.SupplyState[0])
        print('')
               
        startNodes  = []
        endNodes    = []
        flowValues  = []
        flowValues  = []
        
        for startNode,endNode in self.arc_set:
            flow = self.model.flow[(startNode,endNode)].value
            print('Flow on arc %s -> %s: %.2f'%(str(startNode), str(endNode), flow))
            
            startNodes.append(startNode)
            endNodes.append(endNode)
            flowValues.append(flow)

            
        self.flowResults = pd.DataFrame({'start' : startNodes,
                                         'end' : endNodes,
                                         'flow' : flowValues})
        

# --------------------------
# RUN MODEL
# --------------------------

m = waterSimulator()
m.pyomo_model_create()
m.pyomo_model_solve()
m.pyomo_model_print_solutions()