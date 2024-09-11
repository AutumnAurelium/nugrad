#%%
from matplotlib.textpath import TextPath
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from nugrad.value import Value

def visualize_graph(value: Value):
    G = nx.DiGraph()
    
    def build_graph(v: Value):
        #if v not in G.nodes:
        G.add_node(v, label=v.label())
        
        for input in v.inputs:
            G.add_edge(input, v)
            build_graph(input)
    
    build_graph(value)
    
    for i, layer in enumerate(nx.topological_generations(G)):
        for node in layer:
            G.nodes[node]['layer'] = i
    
    pos = nx.multipartite_layout(G, subset_key="layer")

    plt.figure(figsize=(12, 8))
    
    # Calculate node sizes
    node_sizes = []
    prop = fm.FontProperties(size=12)
    for node in G.nodes():
        print(G.nodes[node])
        label = G.nodes[node]['label']
        text_path = TextPath((0, 0), label, prop=prop)
        bbox = text_path.get_extents()
        width, height = bbox.width, bbox.height
        node_sizes.append(width * height * 5)  # Adjust the multiplier as needed
    
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_shape="s", 
            node_size=node_sizes,
            labels={node: G.nodes[node]['label'] for node in G.nodes()})
    
    plt.title("Expression Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    z = x * y
    w = z + x
    v = w * y
    
    v.forward()
    v.backward()

    visualize_graph(v)
# %%
