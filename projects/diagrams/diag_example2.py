import graphviz

# Create a Digraph object
dot = graphviz.Digraph("ProcessFlow", format="png")

# Set global graph attributes
dot.attr(rankdir="LR", size="10,5")

# Define nodes for the main process
dot.node("Start", shape="ellipse", style="filled", fillcolor="lightgreen")
dot.node("Step1", shape="box", style="filled", fillcolor="orange")
dot.node(
    "Decision",
    shape="diamond",
    style="filled",
    fillcolor="lightyellow",
    label="Decision Point?",
)
dot.node("Step2", shape="box", style="filled", fillcolor="orange")
dot.node("End", shape="ellipse", style="filled", fillcolor="lightcoral")

# Define nodes for the optional subprocess
with dot.subgraph(name="cluster_OptionalSubprocess") as sub:
    sub.attr(label="Optional Subprocess", style="filled", color="lightgrey")
    sub.node("SubStep1", shape="box", style="filled", fillcolor="lightcyan")
    sub.node("SubStep2", shape="box", style="filled", fillcolor="lightcyan")
    sub.node("SubStep3", shape="box", style="filled", fillcolor="lightcyan")
    sub.edge("SubStep1", "SubStep2", label="Next")
    sub.edge("SubStep2", "SubStep3", label="Next")

# Define edges for the main process flow
dot.edge("Start", "Step1", label="Begin")
dot.edge("Step1", "Decision", label="Proceed")
dot.edge("Decision", "Step2", label="Yes")
dot.edge("Decision", "End", label="No", style="dashed")

# Integrate the optional subprocess into the main flow
dot.edge("Step2", "SubStep1", label="Optional", style="dotted")
dot.edge("SubStep3", "End", label="Return to Main Flow", style="dotted")

# Render and view the diagram
dot.render("process_flow_diagram", view=True)
