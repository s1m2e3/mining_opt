from ortools.math_opt.python import mathopt

# Build the model.
model = mathopt.Model(name="lp_with_pdlp")

x = model.add_variable(lb=-1.0, ub=1.5, name="x")
y = model.add_variable(lb=0.0,  ub=1.0, name="y")

model.add_linear_constraint(x + y <= 1.5, name="c1")
model.maximize(x + 2 * y)
params = mathopt.SolveParameters(enable_output=True)

result = mathopt.solve(model, mathopt.SolverType.PDLP, params=params)

if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
    raise RuntimeError(f"Did not solve to optimality: {result.termination}")

print("Objective:", result.objective_value())
print("x =", result.variable_values()[x])
print("y =", result.variable_values()[y])
