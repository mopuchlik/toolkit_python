from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment="Process Flow DAG", format="png")

# Define nodes with labels and shapes
dot.node("A", "Start", shape="oval", style="filled", fillcolor="lightgreen")
dot.node("B", "Input 1", shape="box")
dot.node("C", "Input 2", shape="box")
dot.node("D", "Optional Step", shape="diamond", style="filled", fillcolor="lightgrey")
dot.node("E", "Processing Step", shape="box")
dot.node("F", "Output 1", shape="box")
dot.node("G", "Output 2", shape="box")
dot.node("H", "End", shape="oval", style="filled", fillcolor="lightblue")

# Define edges with labels for optional paths
dot.edge("A", "B")
dot.edge("A", "C")
dot.edge("B", "D", label="Optional Path")
dot.edge("C", "E")
dot.edge("D", "E")
dot.edge("E", "F")
dot.edge("E", "G")
dot.edge("F", "H")
dot.edge("G", "H")

# Render the graph to a file
dot.render("process_flow_dag", view=True)
