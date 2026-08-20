from data_classes import Problem
from ortools.linear_solver import pywraplp
import pandas as pd
import datetime
from ortools.math_opt.python import mathopt
import copy
class LG(Problem):
    def __init__(self, data):
        super().__init__(data=data)
        self.data = data
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
    def addVars(self):
        for i in range(len(self.data['blocks'])):
            self.x[i] = self.solver.NumVar(0, 1, f"x[{i}]" )
        print("finished adding vars")
    def addConstraints(self):
        for i in range(len(self.data['blocks'])):
            for j in (self.data['blocks'][i]['depends']):
                self.solver.Add(self.x[i] <= self.x[j])
            print("finished adding constraints for ndoe ", i)
        print("finished adding constraints")
    def addObjective(self):
        objective = self.solver.Objective()
        for i in range(len(self.data['blocks'])):
            objective.SetCoefficient(self.x[i], self.data['blocks'][i]['value']) 
        objective.SetMaximization()
    def writeModel(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()
        
    def solve(self):
        self.solver.EnableOutput()  # before Solve()
        status = self.solver.Solve()
        status_map = {
            pywraplp.Solver.OPTIMAL:        "OPTIMAL",
            pywraplp.Solver.FEASIBLE:       "FEASIBLE",
            pywraplp.Solver.INFEASIBLE:     "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED:      "UNBOUNDED",
            pywraplp.Solver.ABNORMAL:       "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED:     "NOT_SOLVED",
        }
        print("Status code:", status, "->", status_map.get(status, "UNKNOWN"))

        if status == pywraplp.Solver.OPTIMAL:
            obj = self.solver.Objective().Value()
            sol = {i: int(self.x[i].solution_value() + 0.5) for i in self.x}
            for i in range(len(self.data['blocks'])):
                self.data['blocks'][i]['solution']= int(sol[i])
            print("Objective:", obj)
            print("x:", sol)
            return self.data
        elif status == pywraplp.Solver.FEASIBLE:
            print("Feasible (not proven optimal).")
        else:
            print("The solver could not solve the problem.")
        return None

class NPVLG(Problem):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
    def addVars(self):
        for i in self.data['blocks']:
            self.x[i] = {'time':self.solver.NumVar(0,self.data['num_periods'],f"x[{i}][time]"),
                         'tonnage_fraction':self.solver.NumVar(0,self.data['blocks'][i]['tonnage']/self.data['tonnage_scaling'],f"x[{i}][tonnage]")}
        self.x['final_time']=self.solver.NumVar(0,self.data['num_periods'],"x[final_time]")
        print("Finished adding variable node:",i)
    def addConstraints(self):
        for i in self.data['blocks']:
            for j in (self.data['blocks'][i]['depends']):
                self.solver.Add(self.x[i]['time'] >= self.x[j]['time']+(self.x[j]['tonnage_fraction']*self.data['tonnage_scaling'])/self.data['max_mining_rate'])
        for i in self.data['blocks']:
            self.solver.Add(self.x[i]['tonnage_fraction']<=self.data['ramp']*self.x[i]['time'])
            self.solver.Add(self.x['final_time'] >= self.x[i]['time'])
        print("finished adding constraints")            
    def addObjective(self):
        objective = self.solver.Objective()
        for i in self.data['blocks']:
            objective.SetCoefficient(self.x[i]['tonnage_fraction'], self.data['blocks'][i]['value']/self.data['value_scaling'])
        objective.SetCoefficient(self.x['final_time'], -self.data['time_penalty'])
        objective.SetMaximization()
    def writeModel(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()
        
    def solve(self):
        self.solver.EnableOutput()  # before Solve()
        status = self.solver.Solve()
        status_map = {
            pywraplp.Solver.OPTIMAL:        "OPTIMAL",
            pywraplp.Solver.FEASIBLE:       "FEASIBLE",
            pywraplp.Solver.INFEASIBLE:     "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED:      "UNBOUNDED",
            pywraplp.Solver.ABNORMAL:       "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED:     "NOT_SOLVED",
        }
        print("Status code:", status, "->", status_map.get(status, "UNKNOWN"))
        if status == pywraplp.Solver.OPTIMAL:
            obj = self.solver.Objective().Value()
            print("Solution values:")
            for i in self.x:
                if i == 'final_time':
                    continue
                else:
                #  elif self.x[i]['tonnage_fraction'].solution_value() > 0.0:
                    print(f"Block {i}: time={self.x[i]['time'].solution_value()}, tonnage_fraction={self.x[i]['tonnage_fraction'].solution_value()},\
                        mined_tonnage={self.x[i]['tonnage_fraction'].solution_value()*self.data['tonnage_scaling']},max_tonnage={self.data['blocks'][i]['tonnage']}")
            # sol = {i: {"time":self.x[i]['time'].solution_value(), "tonnage_fraction":self.x[i]['tonnage_fraction'].solution_value()} for i in self.x}
            # for i in sol:
            #     print(sol[i]['time'], sol[i]['tonnage_fraction']*self.data['tonnage_scaling'])
            #     self.data['blocks'][i]['solution']= float(sol[i])
            print("Objective:", obj)
            # print("x:", sol)
            return self.data
        elif status == pywraplp.Solver.FEASIBLE:
            print("Feasible (not proven optimal).")
        else:
            print("The solver could not solve the problem.")
        return None

class LowerScheduleMIP(Problem):
    def __init__(self, data):
        super().__init__(data=data)
        self.data = data
        self.T = int(data['num_periods']//data['period_size'])
        self.M = len(data['blocks'])
    def addVars(self):
        # block decision vars
        self.x = {}
        for i in self.data['blocks']:
            self.x[i] = {}
            for k in range(self.T):
                self.x[i][k] = self.model.add_variable(
                    lb=0.0, ub=1.0, is_integer=True, name=f"x_{i}_{k}"
                )

        # centroid vars in separate dict (cleaner than mixing in self.x)
        self.used = {}
        for u in self.data['centroids']:
            self.used[u] = {}
            for k in range(self.T):
                self.used[u][k] = self.model.add_variable(
                    lb=0.0, ub=1.0, is_integer=True, name=f"used_{u}_{k}"
                )

        # interaction vars only once per unordered pair (u < v)
        cent_list = list(self.data['centroids'].keys())
        self.inter = {}
        for a in range(len(cent_list)):
            for b in range(a + 1, len(cent_list)):
                u, v = cent_list[a], cent_list[b]
                self.inter[(u, v)] = {}
                for k in range(self.T):
                    self.inter[(u, v)][k] = self.model.add_variable(
                        lb=0.0, ub=1.0, is_integer=True, name=f"inter_{u}_{v}_{k}"
                    )
    
    def addConstraints(self):
        # mine at most once
        for i in self.data['blocks']:
            self.model.add_linear_constraint(sum(self.x[i][k] for k in range(self.T)) <= 1.0)

        # precedence (your original)
        for i in self.data['blocks']:
            for j in self.data['blocks'][i]['depends']:
                for k in range(1, self.T):
                    self.model.add_linear_constraint(
                        sum(self.x[i][t] for t in range(k + 1))
                        <=
                        sum(self.x[j][t] for t in range(k))
                    )

        # capacity (your original)
        cap = self.data['mining_rate']
        for k in range(self.T):
            self.model.add_linear_constraint(
                sum(self.data['blocks'][i]['tonnage'] * self.x[i][k] for i in self.data['blocks'])
                <= cap
            )

        # centroid used linkage (tight, better than big-M):
        # if any block i in centroid u is mined in k, then used[u,k] must be 1
        for u, block_list in self.data['centroids'].items():
            for k in range(self.T):
                for i in block_list:
                    self.model.add_linear_constraint(self.x[i][k] <= self.used[u][k])

        # interaction linkage: inter(u,v,k) = used(u,k) AND used(v,k)
        for (u, v), interk in self.inter.items():
            for k in range(self.T):
                self.model.add_linear_constraint(interk[k] <= self.used[u][k])
                self.model.add_linear_constraint(interk[k] <= self.used[v][k])
                self.model.add_linear_constraint(interk[k] >= self.used[u][k] + self.used[v][k] - 1)
        
    def addObjective(self):
        beta = 1.0 / (1.0 + self.data['discount_rate'])
        obj = 0.0
        for i in self.data['blocks']:
            v = self.data['blocks'][i]['value']
            for k in range(self.T):
                obj+=self.x[i][k]* (beta**k) * v
        for (u,v),interk in self.inter.items():
            u_set = set(u)
            v_set = set(v)
            intersect = u_set & v_set
            for k in range(self.T):
                obj-= self.inter[(u,v)][k] * (3-len(intersect))*1e5
        self.model.maximize(obj)

    def writeModel(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()
    
    def solve(self, max_time_in_seconds: int = 1800):
        params = mathopt.SolveParameters()
        params.time_limit = datetime.timedelta(seconds=max_time_in_seconds)
        params.enable_output = True
        params.relative_gap_tolerance = 1e-4

        result = mathopt.solve(
            self.model,
            solver_type=getattr(self, "solver_type", mathopt.SolverType.HIGHS),
            params=params,
        )

        # --- Termination info ---
        print("Termination reason:", result.termination.reason)

        if not getattr(result, "solutions", None):
            print("No solutions returned by solver (result.solutions is empty).")
            return None

        var_vals = result.solutions[0].primal_solution.variable_values

        chosen_blocks = {}
        for i in self.data['blocks']:
            for k in range(self.T):
                if var_vals[self.x[i][k]] > 0.5:
                    block_copy = copy.deepcopy(self.data['blocks'][i])
                    block_copy['period'] = k
                    chosen_blocks[i] = block_copy

        try:
            print("Objective:", result.objective_value())
        except Exception:
            print("Objective: unavailable (solver did not report a usable objective value).")

        return chosen_blocks

class NPVLG_Indexed(Problem):
    def __init__(self, data,duals=None):
        super().__init__()
        self.data = data
        self.blocks = list(data['blocks'].keys())
        self.T = int(self.data['num_periods']//self.data['period_size'])
        self.coef = {i: {'tonnage':self.data["blocks"][i]["tonnage"]/self.data['tonnage_scaling'],
                         'value':self.data["blocks"][i]["value"] / self.data['value_scaling'],
                         'inverse_tonnage':self.data['tonnage_scaling']/self.data["blocks"][i]["tonnage"]} for i in self.blocks}
        self.neighbours = self.data['neighbours']
    def addVars(self):
        for i in self.blocks:
            self.x[i] = {}
            for k in range(self.T):
                if self.data['blocks'][i]['min_time']<=k:
                    self.x[i][k]={}
                    self.x[i][k]['tonnage_fraction'] = self.model.add_variable(lb=0.0,ub=self.coef[i]['tonnage'], name = f"x[{i}][{k}][tonnage]")
                    self.x[i][k]['accumulated_tonnage'] = self.model.add_variable(lb=0.0,ub= self.coef[i]['tonnage'], name =f"x[{i}][{k}][accumulated_tonnage]")
                    self.x[i][k]['not_binary_accumulated']= self.model.add_variable(lb=0.0,ub= 10.0, name =f"x[{i}][{k}][distance_accumulated_tonnage]")
                    self.x[i][k]['not_binary_fraction']= self.model.add_variable(lb=0.0,ub= 10.0, name =f"x[{i}][{k}][distance_fraction_tonnage]")
        self.x['theta'] = {}
        for k in range(self.T):
            self.x['theta'][k] = self.model.add_variable(lb=0.0,ub=25.0, name = f"x[theta][{k}]")
        print("finished adding vars")
    def addConstraints(self):
        def has(i, k):
            return (i in self.x) and (k in self.x[i])

        def y(i, k):
            # accumulated_tonnage at (i,k) if it exists, else 0
            if has(i, k):
                return self.x[i][k]['accumulated_tonnage']
            return 0.0

        def u(i, k):
            # tonnage_fraction at (i,k) if it exists, else None
            if has(i, k):
                return self.x[i][k]['tonnage_fraction']
            return None
        self.ctrs["accumulation"] = {}
        # 1) Accumulation constraints: only where states exist
        self.min={}
        for i in self.blocks:
            min_k = self.data['blocks'][i]['min_time']

            # If block is never available within horizon, skip entirely
            if min_k >= self.T:
                continue

            # Base: accumulated at min_time is 0
            # (Since your model has y[i][0]=0; with min_time gating, y[i][min_time]=0)
            self.ctrs["accumulation"][(i,min_k)] = self.model.add_linear_constraint(self.x[i][min_k]['accumulated_tonnage'] == 0.0)
            self.min[i]=min_k
            # Recurrence for k >= min_time: y[k+1] = y[k] + u[k]
            # Only add if both k and k+1 exist (they should if you create from min_time onward)
            for k in range(min_k, self.T - 1):
                if has(i, k) and has(i, k + 1):
                    self.ctrs["accumulation"][(i,k)] =self.model.add_linear_constraint(
                        self.x[i][k + 1]['accumulated_tonnage'] - self.x[i][k]['accumulated_tonnage']-self.x[i][k]['tonnage_fraction']
                        ==  0
                    )
                if has(i,k+1):
                    self.ctrs["accumulation"][(i,k+1)]= self.model.add_linear_constraint(
                        self.x[i][k+1]['accumulated_tonnage'] + self.x[i][k+1]['tonnage_fraction']
                        <=  self.coef[i]['tonnage']
                    )

        print("finished adding accumulation constraints")
        
        # 2) Sequencing / precedence constraints: use y(i,k)=0 when missing
        self.ctrs['precedence'] = {}
        self.ctrs['precedence_linking'] = {}
        for i in self.blocks:
            for j in self.data['blocks'][i]['depends']:
                for k in range(self.T):
                    if has(i, k) and has(j, k):
                        self.ctrs["precedence"][(i,j,k)] = self.model.add_linear_constraint(
                            y(i, k) * self.coef[i]['inverse_tonnage']-y(j, k) * self.coef[j]['inverse_tonnage']
                            <=0
                        )
                        self.ctrs["precedence_linking"][(i,j,k)] = self.model.add_linear_constraint(
                            u(i, k)-y(j, k) * ((self.coef[i]['tonnage']) / self.coef[j]['tonnage'])
                            <=0
                        )
        
        print("finished adding precedence constraints")
        # 3) Capacity constraints: sum only existing u(i,k)
        cap = self.data['mining_rate'] / self.data['tonnage_scaling']
        self.ctrs['capacity'] = {}
        for k in range(self.T):
            terms = []
            for i in self.blocks:
                uk = u(i, k)
                if uk is not None:
                    terms.append(uk)
            self.ctrs['capacity'][k] = self.model.add_linear_constraint(sum(terms) <= cap)
        
        self.ctrs['feasibility'] = {}
        self.ctrs['optimality'] = {}
        # 4) Not Binary constraints
        self.ctrs['not_binary_accumulated_less_than'] = {}
        self.ctrs['not_binary_accumulated_greater_than'] = {}
        self.ctrs['not_binary_fraction_less_than'] = {}
        self.ctrs['not_binary_fraction_greater_than'] = {}
        for i in self.blocks:
            for k in range(self.T):
                if has(i, k):
                    self.ctrs['not_binary_accumulated_less_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_accumulated'] >= y(i,k)*(1.0/ self.coef[i]['tonnage'])-0.5)
                    self.ctrs['not_binary_accumulated_greater_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_accumulated'] >= 0.5 - y(i,k)*(1.0/ self.coef[i]['tonnage']))
                    self.ctrs['not_binary_fraction_less_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_fraction'] >= u(i,k)*(1.0/ self.coef[i]['tonnage'])-0.5)
                    self.ctrs['not_binary_fraction_greater_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_fraction'] >= 0.5 - u(i,k)*(1.0/ self.coef[i]['tonnage']))
        print("finished adding binary constraints ")
        print("finished adding constraints")
    def addDualConstraints(self,solutions,iteration):
        for k in solutions:
            if solutions[k]['status'].termination.reason == mathopt.TerminationReason.INFEASIBLE:
                self.addFeasibilityconstraints(solutions[k],k,iteration)
            elif solutions[k]['status'].termination.reason == mathopt.TerminationReason.OPTIMAL:
                self.addOptimalityConstraints(solutions[k],k,iteration)
    def addFeasibilityconstraints(self,solution,k,iteration):
        rays = solution['result']
        max_connection = max([max(abs(rays['connection'][i]['start']),abs(rays['connection'][i]['end'])) for i in rays['connection']])
        max_capacity = max(abs(rays['capacity'][m]) for m in rays['capacity'])
        max_scale = max(max_connection,max_capacity)
        lhs = sum((rays['connection'][i]['start']+rays['connection'][i]['end'])*self.x[i][k]['accumulated_tonnage'] \
            + rays['connection'][i]['end']*self.x[i][k]['tonnage_fraction'] for i in rays['connection'])
        rhs = sum(rays['capacity'][m] for m in rays['capacity'])*rays['meta']['mining_rate']
        self.ctrs['feasibility'][(k,iteration)] = self.model.add_linear_constraint(lhs/max_scale*10 >= -rhs/max_scale*10)
    def addOptimalityConstraints(self,solution,k,iteration):
        duals = solution['result']["duals"]
        lhs = self.x['theta'][k]
        rhs = duals["Qbar"]
        rhs += sum(duals["connection"][i]["start"]*(self.x[i][k]['accumulated_tonnage']-duals["connection"][i]["accumulated_tonnage"]) for i in duals["connection"])+\
            sum(duals["connection"][i]["end"]*((self.x[i][k]['accumulated_tonnage']+self.x[i][k]['tonnage_fraction'])-(duals["connection"][i]["accumulated_tonnage"]+duals["connection"][i]["tonnage_fraction"])) for i in duals["connection"])
        self.ctrs['optimality'][(k,iteration)] = self.model.add_linear_constraint(lhs <= rhs)
        
    def addObjective(self):
        beta = 1.0 / (1.0 + self.data['discount_rate'])
        obj = 0
        for i in self.blocks:
            v = self.data['blocks'][i]['value'] / self.data['value_scaling']
            for k in range(self.T):
                if (i in self.x) and (k in self.x[i]):
                    obj += self.x[i][k]['tonnage_fraction']* (beta ** k) * v
                    obj += (self.x[i][k]['not_binary_accumulated']+self.x[i][k]['not_binary_fraction']/2)*self.data['binary_penalty']
        for i in self.neighbours:
            for j in self.neighbours[i]:
                if i!=j and j not in self.data['blocks'][i]['depends'] and i not in self.data['blocks'][j]['depends']:
                    for k in range(self.T):
                        if (i in self.x) and (j in self.x) and (k in self.x[i]) and (k in self.x[j]):
                            obj += (self.x[i][k]['tonnage_fraction']*(self.coef[i]['inverse_tonnage'])+self.x[j][k]['tonnage_fraction']*(self.coef[j]['inverse_tonnage']))*self.data['binary_penalty']*0.1
            
        self.model.maximize(obj)
        
    def addObjectiveWithDual(self):
        beta = 1.0 / (1.0 + self.data['discount_rate'])
        obj = 0
        for i in self.blocks:
            v = self.data['blocks'][i]['value'] / self.data['value_scaling']
            for k in range(self.T):
                if (i in self.x) and (k in self.x[i]):
                    obj += self.x[i][k]['tonnage_fraction']*(1.0/ self.coef[i]['tonnage'])* (beta ** k) * v
                    obj += (self.x[i][k]['not_binary_accumulated']+self.x[i][k]['not_binary_fraction']/2)*self.data['binary_penalty']
        for i in self.neighbours:
            for j in self.neighbours[i]:
                if i!=j and j not in self.data['blocks'][i]['depends'] and i not in self.data['blocks'][j]['depends']:
                    for k in range(self.T):
                        if (i in self.x) and (j in self.x) and (k in self.x[i]) and (k in self.x[j]):
                            obj += (self.x[i][k]['tonnage_fraction']*(self.coef[i]['inverse_tonnage'])+self.x[j][k]['tonnage_fraction']*(self.coef[j]['inverse_tonnage']))*self.data['binary_penalty']*0.1
        for k in range(self.T):
            obj += self.x['theta'][k]
        self.model.maximize(obj)

    def writeModel(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()
    
    def solve(self, max_time_in_seconds: int = 600):
        params = mathopt.SolveParameters()
        
        params.time_limit = datetime.timedelta(seconds=max_time_in_seconds)
        params.enable_output = True
        params.relative_gap_tolerance = 1e-4
        
        result = mathopt.solve(
            self.model,
            solver_type=getattr(self, "solver_type", mathopt.SolverType.HIGHS),
            params=params,
        )

        # --- Termination info ---
        print("Termination reason:", result.termination.reason)

        # MathOpt may return 0+ solutions even if not optimal/feasible.
        if not getattr(result, "solutions", None):
            print("No solutions returned by solver (result.solutions is empty).")
            return result

        # Take the first solution (typically the best one returned).
        sol = result.solutions[0]
        
        # -------------------------
        # PRIMAL: print last returned variable values (even if infeasible)
        # -------------------------
        if sol.primal_solution is None:
            print("No primal solution object returned.")
        else:
            ps = sol.primal_solution
            # The solver's *claimed* feasibility status (FEASIBLE / INFEASIBLE / UNDETERMINED)
            # Field name varies slightly by OR-Tools version; try common ones.
            feas = getattr(ps, "feasibility_status", None) or getattr(ps, "status", None)
            if feas is not None:
                print("Primal feasibility status (claimed):", feas)

            var_vals = ps.variable_values
            # Objective value: available in some versions as result.objective_value()
            # only when primal feasible; otherwise it may not exist / may be meaningless.
            try:
                print("Objective (from result, if available):", result.objective_value())
            except Exception:
                # If not available, you can still compute it yourself if you stored objective expression,
                # but MathOpt does not always expose objective evaluation for infeasible solutions.
                print("Objective: unavailable (solver did not report a usable objective value).")

            print("Solution values (last returned primal solution):")
            
            # Helper inside solve (or use your existing has() logic)
            def has_var(i, k):
                return (i in self.x) and (i != "theta") and isinstance(self.x[i], dict) and (k in self.x[i])

            chosen_blocks = {
            "nodes": {},
            "meta": {
                "tonnage_scaling": self.data["tonnage_scaling"],
                "value_scaling": self.data["value_scaling"],
                "mining_rate": self.data["mining_rate"],
                "discount_rate": self.data["discount_rate"],
                "period_size": self.data["period_size"],
                "binary_penalty": self.data["binary_penalty"],
                "neighbours": self.data["neighbours"]
                    },
                }
            
            for i in self.blocks:  # iterate real blocks, not self.x (self.x includes 'theta')
                for k in range(self.T):
                    if not has_var(i, k):
                        continue
                    u_var = self.x[i][k].get("tonnage_fraction")
                    y_var = self.x[i][k].get("accumulated_tonnage")
                    if u_var is not None and y_var is not None:
                        # Safety: only read values if var exists
                        u_val = var_vals[u_var]
                        y_val = var_vals[y_var]
                    else:
                        continue
                    if (u_val > 0.0) or (y_val > 0.0):
                        chosen_blocks["nodes"].setdefault(i, {})

                        # IMPORTANT: copy the block dict so we don't mutate self.data
                        block_copy = copy.deepcopy(self.data["blocks"][i])

                        block_copy["tonnage_fraction"] = float(u_val)
                        block_copy["accumulated_tonnage"] = float(y_val)

                        chosen_blocks["nodes"][i][k] = block_copy

                        # Also record dependencies at the same k (only if their vars exist)
                        for depend in self.data["blocks"][i].get("depends", []):
                            if not has_var(depend, k):
                                continue

                            chosen_blocks["nodes"].setdefault(depend, {})

                            u_d_var = self.x[depend][k].get("tonnage_fraction")
                            y_d_var = self.x[depend][k].get("accumulated_tonnage")

                            u_d_val = var_vals[u_d_var] if u_d_var is not None else 0.0
                            y_d_val = var_vals[y_d_var] if y_d_var is not None else 0.0

                            dep_copy = copy.deepcopy(self.data["blocks"][depend])
                            dep_copy["tonnage_fraction"] = float(u_d_val)
                            dep_copy["accumulated_tonnage"] = float(y_d_val)

                            chosen_blocks["nodes"][depend][k] = dep_copy

        # -------------------------
        # DUAL: print last returned duals if present (even if infeasible)
        # -------------------------
        if sol.dual_solution is None:
            print("No dual solution object returned (duals unavailable).")
        else:
            ds = sol.dual_solution
            dfeas = getattr(ds, "feasibility_status", None) or getattr(ds, "status", None)
            if dfeas is not None:
                print("Dual feasibility status (claimed):", dfeas)

            dual_vals = ds.dual_values
            print("Constraint duals (last returned dual solution):")
       
        return result, chosen_blocks, result.objective_value()

class NPVLG_Intra_Period(Problem):
    def __init__(self, block_data,meta_data,periods,k):
        super().__init__()
        self.solver_type = mathopt.SolverType.PDLP
        self.block_data = copy.deepcopy(block_data)
        self.meta_data = copy.deepcopy(meta_data)
        self.T = periods
        self.k = k
        self.coef = {i: {'tonnage':self.block_data[i]["tonnage"]/self.meta_data['tonnage_scaling'],
                         'value':self.block_data[i]["value"] / self.meta_data['value_scaling']} for i in self.block_data}
        self.meta_data['discount_rate'] = (1+self.meta_data['discount_rate'])**(1/self.T)-1
        self.meta_data['mining_rate'] = self.meta_data['mining_rate']/self.T
        
    def addVars(self):
        print("Total nodes:", len(self.block_data), " in period:", self.k)
        for i in self.block_data:
            self.x[i] = {self.T:{}}
            for k in range(self.T):
                self.x[i][k]={}
                self.x[i][k]['tonnage_fraction'] = self.model.add_variable(lb=0.0,ub=self.coef[i]['tonnage'], name = f"x[{i}][{k}][tonnage]")
                self.x[i][k]['accumulated_tonnage'] = self.model.add_variable(lb=0.0,ub= self.coef[i]['tonnage'], name =f"x[{i}][{k}][accumulated_tonnage]")
                self.x[i][k]['not_binary_accumulated'] = self.model.add_variable(lb=0.0,ub= 0.5, name =f"x[{i}][{k}][distance_accumulated_tonnage]")
                self.x[i][k]['not_binary_fraction'] = self.model.add_variable(lb=0.0,ub= 0.5, name =f"x[{i}][{k}][distance_fraction_tonnage]")
            self.x[i][self.T]['accumulated_tonnage'] = self.model.add_variable(lb=0.0,ub= self.coef[i]['tonnage'], name =f"x[{i}][{self.T}][accumulated_tonnage]")
        print("finished adding vars")
    def addConstraints(self):
        
        def has(i, k):
            return (i in self.x) and (k in self.x[i])

        def y(i, k):
            # accumulated_tonnage at (i,k) if it exists, else 0
            if has(i, k):
                return self.x[i][k]['accumulated_tonnage']
            return 0.0

        def u(i, k):
            # tonnage_fraction at (i,k) if it exists, else None
            if has(i, k):
                return self.x[i][k]['tonnage_fraction']
            return None
        self.ctrs["accumulation"] = {}
        # 1) Accumulation constraints: only where states exist
        for i in self.block_data:
            # Recurrence for k >= min_time: y[k+1] = y[k] + u[k]
            # Only add if both k and k+1 exist (they should if you create from min_time onward)
            for k in range(0, self.T):
                if has(i, k) and has(i, k + 1):
                    self.ctrs["accumulation"][(i,k)] =self.model.add_linear_constraint(
                        self.x[i][k + 1]['accumulated_tonnage'] - self.x[i][k]['accumulated_tonnage']-self.x[i][k]['tonnage_fraction']
                        ==  0
                    )
        print("finished adding accumulation constraints")
        self.ctrs["connection"] = {}
        for i in self.block_data:
            self.ctrs["connection"][(i,"start")] = self.model.add_linear_constraint(
                self.x[i][0]['accumulated_tonnage'] == self.block_data[i]['accumulated_tonnage']
            )
            self.ctrs["connection"][(i,"end")] = self.model.add_linear_constraint(
                self.x[i][self.T]['accumulated_tonnage'] == self.block_data[i]['accumulated_tonnage']+self.block_data[i]['tonnage_fraction']
            )
            
        # 2) Sequencing / precedence constraints: use y(i,k)=0 when missing
        self.ctrs['precedence'] = {}
        for i in self.block_data:
            for j in self.block_data[i]['depends']:
                for k in range(self.T):
                    if has(i, k) and has(j, k):
                        self.ctrs["precedence"][(i,j,k)] = self.model.add_linear_constraint(
                            y(i, k) * (1.0 / self.coef[i]['tonnage'])-y(j, k) * (1.0 / self.coef[j]['tonnage'])
                            <=0
                        )
        print("finished adding precedence constraints")

        # 3) Capacity constraints: sum only existing u(i,k)
        cap = self.meta_data['mining_rate'] / self.meta_data['tonnage_scaling']
        self.ctrs['capacity'] = {}
        for k in range(self.T):
            terms = []
            for i in self.block_data:
                uk = u(i, k)
                if uk is not None:
                    terms.append(uk)
            self.ctrs['capacity'][k] = self.model.add_linear_constraint(sum(terms) <= cap)
        print("finished adding capacity constraints ")
        # 4) Not Binary constraints
        self.ctrs['not_binary_accumulated_less_than'] = {}
        self.ctrs['not_binary_accumulated_greater_than'] = {}
        self.ctrs['not_binary_fraction_less_than'] = {}
        self.ctrs['not_binary_fraction_greater_than'] = {}
        for i in self.block_data:
            for k in range(self.T):
                if has(i, k):
                    self.ctrs['not_binary_accumulated_less_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_accumulated'] >= y(i,k)*(1.0/ self.coef[i]['tonnage'])-0.5)
                    self.ctrs['not_binary_accumulated_greater_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_accumulated'] >= 0.5 - y(i,k)*(1.0/ self.coef[i]['tonnage']))
                    self.ctrs['not_binary_fraction_less_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_fraction'] >= u(i,k)*(1.0/ self.coef[i]['tonnage'])-0.5)
                    self.ctrs['not_binary_fraction_greater_than'][(i,k)] = self.model.add_linear_constraint(self.x[i][k]['not_binary_fraction'] >= 0.5 - u(i,k)*(1.0/ self.coef[i]['tonnage']))
        print("finished adding binary constraints ")
        print("finished adding constraints")

    def addObjective(self):
        beta = 1.0 / (1.0 + self.meta_data['discount_rate'])
        obj = 0
        for i in self.block_data:
            v = self.block_data[i]['value'] / self.meta_data['value_scaling']
            for k in range(self.T):
                if (i in self.x) and (k in self.x[i]):
                    obj += self.x[i][k]['tonnage_fraction']*(1.0/ self.coef[i]['tonnage'])*(beta ** k) * v
                    obj += (self.x[i][k]['not_binary_accumulated']+self.x[i][k]['not_binary_fraction']/2)*self.meta_data['binary_penalty']
        for i in self.block_data:
            for j in self.meta_data['neighbours'][i]:
                if j in self.block_data and i!=j:
                    if  j not in self.block_data[i]['depends'] and i not in self.block_data[j]['depends']:
                        for k in range(self.T):
                            if (i in self.x) and (j in self.x) and (k in self.x[i]) and (k in self.x[j]):
                                obj += (self.x[i][k]['tonnage_fraction']*(1.0/ self.coef[i]['tonnage'])+self.x[j][k]['tonnage_fraction']*(1.0/ self.coef[i]['tonnage']))*self.meta_data['binary_penalty']
       
        self.model.maximize(obj)

    def write_model(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()
    
    def solve(self, max_time_in_seconds: int = 600):
        params = mathopt.SolveParameters()
        params.time_limit = datetime.timedelta(seconds=max_time_in_seconds)
        params.pdlp.termination_criteria.eps_primal_infeasible = 1e-4
        params.pdlp.termination_criteria.eps_dual_infeasible   = 1e-4
        params.pdlp.presolve_options.use_glop = True
        params.enable_output = True
        params.relative_gap_tolerance = 1e-4
        result = mathopt.solve(
            self.model,
            solver_type=getattr(self, "solver_type", mathopt.SolverType.PDLP),
            params=params,
        )
        
        # --- Termination info ---
        print("Termination reason:", result.termination.reason)
        if result.termination.reason in [mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED,mathopt.TerminationReason.INFEASIBLE, mathopt.TerminationReason.NO_SOLUTION_FOUND]:
            if not result.has_dual_ray():
                print("Solver declared infeasible but did not return a dual ray.")
                print(result.termination.reason)
                input('yipo')
                return result, {}
            y = result.ray_dual_values()        # dict: LinearConstraint -> float
            rays = {'connection':{},'capacity':{},'meta':{}}
            for i in self.block_data:
                rays['connection'][i] = {'start':y.get(self.ctrs["connection"][(i, "start")], 0.0),'end':y.get(self.ctrs["connection"][(i, "end")], 0.0)}
            for m in range(self.T):
                rays['capacity'][m] = y.get(self.ctrs['capacity'][m], 0.0)
            rays['meta']['mining_rate']=self.meta_data['mining_rate'] / self.meta_data['tonnage_scaling']
            
            return result, rays         
        elif result.termination.reason == mathopt.TerminationReason.NUMERICAL_ERROR:
            return result, {}
        else:
            # MathOpt may return 0+ solutions even if not optimal/feasible.
            if not getattr(result, "solutions", None):
                print("No solutions returned by solver (result.solutions is empty).")
                return result,None
            
            # Take the first solution (typically the best one returned).
            sol = result.solutions[0]
            # -------------------------
            # PRIMAL: print last returned variable values (even if infeasible)
            # -------------------------
            if sol.primal_solution is None:
                print("No primal solution object returned.")
            else:
                ps = sol.primal_solution
                # The solver's *claimed* feasibility status (FEASIBLE / INFEASIBLE / UNDETERMINED)
                # Field name varies slightly by OR-Tools version; try common ones.
                feas = getattr(ps, "feasibility_status", None) or getattr(ps, "status", None)
                if feas is not None:
                    print("Primal feasibility status (claimed):", feas)

                var_vals = ps.variable_values

                # Objective value: available in some versions as result.objective_value()
                # only when primal feasible; otherwise it may not exist / may be meaningless.
                try:
                    print("Objective (from result, if available):", result.objective_value())
                except Exception:
                    # If not available, you can still compute it yourself if you stored objective expression,
                    # but MathOpt does not always expose objective evaluation for infeasible solutions.
                    print("Objective: unavailable (solver did not report a usable objective value).")
                # Helper inside solve (or use your existing has() logic)
                def has_var(i, k):
                    return (i in self.x) and isinstance(self.x[i], dict) and (k in self.x[i])

                print("Solution values (last returned primal solution):")
                chosen_blocks = {'nodes':{}}
                for i in self.block_data:  # iterate real blocks, not self.x (self.x includes 'theta')
                    for k in range(self.T):
                        if not has_var(i, k):
                            continue
                        u_var = self.x[i][k].get("tonnage_fraction")
                        y_var = self.x[i][k].get("accumulated_tonnage")
                        if u_var is not None and y_var is not None:
                            # Safety: only read values if var exists
                            u_val = var_vals[u_var]
                            y_val = var_vals[y_var]
                        else:
                            continue
                        if (u_val > 0.0) or (y_val > 0.0):
                            chosen_blocks["nodes"].setdefault(i, {})

                            # IMPORTANT: copy the block dict so we don't mutate self.data
                            block_copy = copy.deepcopy(self.block_data[i])

                            block_copy["tonnage_fraction"] = float(u_val)*(1.0/ self.coef[i]['tonnage'])
                            block_copy["accumulated_tonnage"] = float(y_val)*(1.0/ self.coef[i]['tonnage'])

                            chosen_blocks["nodes"][i][k] = block_copy
                            # Also record dependencies at the same k (only if their vars exist)
                            for depend in self.block_data[i].get("depends", []):
                                if not has_var(depend, k):
                                    continue

                                chosen_blocks["nodes"].setdefault(depend, {})

                                u_d_var = self.x[depend][k].get("tonnage_fraction")
                                y_d_var = self.x[depend][k].get("accumulated_tonnage")

                                u_d_val = var_vals[u_d_var] if u_d_var is not None else 0.0
                                y_d_val = var_vals[y_d_var] if y_d_var is not None else 0.0

                                dep_copy = copy.deepcopy(self.block_data[depend])
                                dep_copy["tonnage_fraction"] = float(u_d_val)*(1.0/ self.coef[i]['tonnage'])
                                dep_copy["accumulated_tonnage"] = float(y_d_val)*(1.0/ self.coef[i]['tonnage'])

                                chosen_blocks["nodes"][depend][k] = dep_copy

                # -------------------------
                if sol.dual_solution is None:
                    print("No dual solution object returned (duals unavailable).")
                    return result, None
                else:
                    ds = sol.dual_solution
                    dfeas = getattr(ds, "feasibility_status", None) or getattr(ds, "status", None)
                    if dfeas is not None:
                        print("Dual feasibility status (claimed):", dfeas)

                    dual_vals = ds.dual_values
                    print("Constraint duals (last returned dual solution):")
                    
                    duals = {
                    "Qbar": ps.objective_value}
                    connection = {
                    i:{"start": dual_vals[self.ctrs['connection'][(i,"start")]],
                        "end":   dual_vals[self.ctrs['connection'][(i,"end")]],
                        "accumulated_tonnage":     self.block_data[i]["accumulated_tonnage"],
                        "tonnage_fraction":     self.block_data[i]["tonnage_fraction"]}
                    for i in self.block_data}
                    duals["connection"] = connection
                    
                    return result,{'primal_solution':chosen_blocks,'duals':duals}
        

class NPVLG_Hierarchical(Problem):
    def __init__(self, data,periods_per_period):
        super().__init__()
        self.top_model = NPVLG_Indexed(data)
        self.periods_per_period = periods_per_period
    def solve(self, max_iters = 100, dual_gap=1e-4):
        self.top_model.writeModel()
        _, chosen_blocks, _ = self.top_model.solve()
        ub = 1e6
        final_solution = {}
        for i in range(max_iters):
            subproblems = {}
            for k in range(self.top_model.T):
                relevant_blocks =  {node: chosen_blocks['nodes'][node][k] for node in chosen_blocks['nodes'] if k in chosen_blocks['nodes'][node]}
                subproblems[k] = NPVLG_Intra_Period(block_data=relevant_blocks,meta_data=chosen_blocks['meta'],periods=self.periods_per_period,k=k)
            solutions = {}
            all_optimal = True
            for k in range(self.top_model.T):
                subproblems[k].write_model()
                status, solution = subproblems[k].solve()
                solutions[k] = {'status':status,'result':solution}
                all_optimal = all_optimal and status.termination.reason == mathopt.TerminationReason.OPTIMAL
            self.top_model.addDualConstraints(solutions,i)
            if i == 0:
                self.top_model.addObjectiveWithDual()
            _, chosen_blocks,value =self.top_model.solve()
            if ub - value < dual_gap:
                break
            elif all_optimal:
                print("New ub is",ub,"new value is",value)
                ub = value
        for k in range(self.top_model.T):
            if 'primal_solution' in solutions[k]['result']:
                final_solution[k] = solutions[k]['result']['primal_solution']['nodes']
            else:
                print("No feasible solution for period",k)
        return final_solution

class PushbackScheduleMIP(Problem):
    """
    MIP counterpart to the pushback-sequencing simulator/neural-net pipeline
    (simulator.py/train.py). Ingests the same pushbacks.csv data (see
    lg_utils.load_pushback_schedule_data) and produces an exact,
    period-indexed schedule of pushbacks, instead of a learned policy's
    greedy/beam-search sequence.

    A pushback need not be completed within a single period: z[p] is a
    binary "this depth level is the one committed to for its row", and
    x[p][k] is the continuous fraction of it completed in period k -
    income, cost, and block-count all scale linearly with completed
    fraction, so a pushback too large for one period's remaining capacity
    spills over into a later one instead of being dropped outright. See
    lg_utils.py's "Pushback Sequencing Problem" docstring for the full,
    canonical definition this and beam_search_pushback_schedule both solve.
    """
    def __init__(self, data):
        super().__init__(data=data)
        self.data = data
        self.T = int(data['num_periods'] // data['period_size'])
        self.pids = list(data['pushbacks'].keys())

    def addVars(self):
        self.z = {}
        self.x = {}
        for p in self.pids:
            self.z[p] = self.model.add_variable(lb=0.0, ub=1.0, is_integer=True, name=f"z_{p}")
            self.x[p] = {}
            for k in range(self.T):
                self.x[p][k] = self.model.add_variable(
                    lb=0.0, ub=1.0, is_integer=False, name=f"x_{p}_{k}"
                )
        print("finished adding vars")

    def addConstraints(self):
        for p in self.pids:
            # can't complete more than 100% in total ...
            self.model.add_linear_constraint(sum(self.x[p][k] for k in range(self.T)) <= 1.0)
            # ... and can't make any progress at all unless selected
            for k in range(self.T):
                self.model.add_linear_constraint(self.x[p][k] <= self.z[p])

        # at most one depth level *selected* per (x,y,z) row - a deeper
        # level's footprint already contains the shallower ones at the
        # same row, so they are mutually exclusive alternatives
        for pids in self.data['row_groups']:
            if len(pids) > 1:
                self.model.add_linear_constraint(sum(self.z[p] for p in pids) <= 1.0)
        print("finished adding row-exclusivity constraints")

        # precedence (true per-block rule, lifted to the pushback level):
        # each distinct external predecessor block p depends on is its own
        # independent requirement (AND across groups) - p's cumulative
        # completed fraction through period k cannot exceed the cumulative
        # completed fraction, through period k-1, of whichever pushback(s)
        # could supply that specific block (OR within a group - see
        # lg_utils.load_pushback_schedule_data/compute_block_precedence)
        for p in self.pids:
            groups = self.data['pushback_predecessor_groups'].get(p)
            if not groups:
                continue
            for k in range(1, self.T):
                lhs = sum(self.x[p][t] for t in range(k + 1))
                for group in groups:
                    rhs = sum(self.x[q][t] for q in group for t in range(k))
                    self.model.add_linear_constraint(lhs <= rhs)
        print("finished adding precedence constraints")

        # true tonnage capacity per period (weighted by completed fraction
        # that period)
        cap = self.data.get('tonnage_capacity_per_period')
        if cap:
            for k in range(self.T):
                self.model.add_linear_constraint(
                    sum(self.data['pushbacks'][p]['tonnage'] * self.x[p][k] for p in self.pids)
                    <= cap
                )
            print("finished adding capacity constraints")

        # a grid-block shared by multiple pushbacks can be claimed by at
        # most one *selected* pushback, regardless of completion fraction
        for b, owners in self.data['block_owner'].items():
            self.model.add_linear_constraint(sum(self.z[p] for p in owners) <= 1.0)
        print("finished adding block-overlap constraints")
        print("finished adding constraints")

    def addObjective(self):
        beta = 1.0 / (1.0 + self.data['discount_rate'])
        obj = 0.0
        for p in self.pids:
            # 'cost' is stored as an already-negative quantity, so combine
            # with addition, not subtraction
            v = self.data['pushbacks'][p]['income'] + self.data['pushbacks'][p]['cost']
            for k in range(self.T):
                obj += self.x[p][k] * (beta ** k) * v
        self.model.maximize(obj)

    def writeModel(self):
        self.addVars()
        self.addConstraints()
        self.addObjective()

    def solve(self, max_time_in_seconds: int = 1800):
        params = mathopt.SolveParameters()
        params.time_limit = datetime.timedelta(seconds=max_time_in_seconds)
        params.enable_output = True
        params.relative_gap_tolerance = 1e-4

        result = mathopt.solve(
            self.model,
            solver_type=getattr(self, "solver_type", mathopt.SolverType.HIGHS),
            params=params,
        )

        print("Termination reason:", result.termination.reason)

        if not getattr(result, "solutions", None):
            print("No solutions returned by solver (result.solutions is empty).")
            return None

        var_vals = result.solutions[0].primal_solution.variable_values

        chosen = {}
        for p in self.pids:
            schedule = {k: var_vals[self.x[p][k]] for k in range(self.T) if var_vals[self.x[p][k]] > 1e-6}
            if schedule:
                entry = copy.deepcopy(self.data['pushbacks'][p])
                entry['schedule'] = schedule                       # period -> fraction completed
                entry['first_period'] = min(schedule)
                entry['completed_fraction'] = sum(schedule.values())
                chosen[p] = entry

        # explicit mining order: by first period worked, then by
        # descending net value as a tie-break
        order = sorted(
            chosen.keys(),
            key=lambda p: (chosen[p]['first_period'], -(chosen[p]['income'] + chosen[p]['cost']))
        )

        try:
            objective_value = result.objective_value()
        except Exception:
            objective_value = None
        print("Objective:", objective_value)

        return {'chosen': chosen, 'order': order, 'objective_value': objective_value}