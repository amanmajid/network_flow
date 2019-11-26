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

    def __init__(self, node_file, arc_file, supply_demand_file):

        # Read in the node_data
        self.node_data = pd.read_csv(node_file)
        self.node_data.set_index(['Node'], inplace=True)
        self.node_data.sort_index(inplace=True)

        # Read in the arc_data
        self.arc_data = pd.read_csv(arc_file)
        self.arc_data.set_index(['Start','End'], inplace=True)
        self.arc_data.sort_index(inplace=True)

        # Read in the supply_demand_data
        self.supply_demand_data = pd.read_csv(supply_demand_file)
        self.supply_demand_data.set_index(['Timestep'], inplace=True)
        self.supply_demand_data.sort_index(inplace=True)
        
        self.node_set = self.node_data.index.unique()
        self.arc_set = self.arc_data.index.unique()
        self.create_pyomo_model()
        
    def create_pyomo_model(self):

        model           = pe.ConcreteModel()
        model.dual      = pe.Suffix( direction=pe.Suffix.IMPORT )

        # ------
        # Sets
        model.nodes = pe.Set( initialize=self.node_set )
        model.arcs  = pe.Set( initialize=self.arc_set, dimen=2 )

        # ------
        # Params
        model.sourceNodes   = pe.Param(within=model.nodes)
        model.sinkNodes     = pe.Param(within=model.nodes)
        model.capacity      = pe.Param(model.arcs, mutable=False)
        model.upperBound    = pe.Param(model.arcs, mutable=False)
        model.lowerBound    = pe.Param(model.arcs, mutable=False)
        model.cost          = pe.Param(model.arcs, mutable=False)

        # ------
        # Vars
        model.flow = pe.Var(model.arcs, within=pe.NonNegativeReals)     # The flow over each arc

        # --------------------------
        # OPTIMISATION
        # --------------------------

        # ------
        # Objective function

        # Minimise total cost of flow
        def objective_function(model):
            return sum(model.cost[i,j] * model.flow[i,j] for (i,j) in model.arcs)

        model.totalCost = pe.Objective(rule=objective_function, sense=pe.minimize)

        # ------
        # Constraints

        # Enforce upper bound to arc flow
        def rule_arcFlow_upper(model, i, j):
            return model.flow[i,j] <= model.upperBound[i,j]

        model.arc_upper_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_upper)

        # Enforce lower bound to arc flow
        def rule_arcFlow_lower(model, i, j):
            return model.flow[i,j] >= model.lowerBound[i,j]

        model.arc_lower_limit = pe.Constraint(model.arcs, rule=rule_arcFlow_lower)

        # Enforce flow through each node (mass balance)
        def rule_mass_balance(model, node):
            if node == model.sourceNodes or node == model.sinkNodes:
                return pe.Constraint.Skip
            inFlow   = sum(model.flow[i,j] for (i,j) in model.arcs if j == node)
            outFlow  = sum(model.flow[i,j] for (i,j) in model.arcs if i == node)
            return inFlow == outFlow

        model.flow = pe.Constraint(model.nodes, rule=rule_mass_balance)
        
        self.model = model

    # --------------------------
    # SOLVING 
    # --------------------------

    def solve_model(self, solver='gurobi', tee=False):
        solver = pyomo.opt.SolverFactory(solver)

        print('-----Solving Pyomo Model-----')
        self.results = solver.solve(self.model)

        # Check that we actually computed an optimal solution, load results
        if (self.results.solver.status != pyomo.opt.SolverStatus.ok):
            print('Check solver not ok?')
        if (self.results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            print('Check solver optimality?')


# --------------------------
# Run model
# --------------------------

m = waterSimulator('../data/nodes.csv', '../data/arcs.csv', '../data/supply_demand.csv')
m.solve_model(solver='gurobi')